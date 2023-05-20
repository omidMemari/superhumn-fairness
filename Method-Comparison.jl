using Statistics
using JuMP
import Gurobi, HiGHS, SCIP, QuadraticToBinary, Ipopt
using DataFrames, CSV
import Random
using LinearAlgebra
using StatsBase
# set global parameters
global_constraint_tol = 0.00001

function train_test_split(df, test_size, seed)
    n = nrow(df)
    Random.seed!(seed)
    ix = Random.shuffle(1:n)
    df_shuffled = df[ix,:]
    train_ix = Int(floor(n*(1-test_size)))
    train = df_shuffled[1:train_ix,:]
    test = df_shuffled[(train_ix+1):n,:]
    return train, test
end

function get_parameters(df, G, B, scale = true, scale_param = 1)
    if scale
        N_df = nrow(df) # normalizing parameter to avoid huge numbers 
    else
        N_df = 1
    end
    # define additional parameters for the problem 
    S = zeros((G,B)) # average scores in each bin (for minimum movement objective)
    N = zeros((G,B)) # normalized to avoid numeric issues
    Y = zeros((G,B)) # normalized to avoid numeric issues 
    for b=1:B
        for g=1:G
            sub_df = df[(df[:,"g"].==g).&(df[:,"bin"].==b),["y","s"]]
            if nrow(sub_df) == 0
                println("Empty bin for group $g bin $b")
                N[g,b] = S[g,b] = Y[g,b] = 0
            else
                N[g,b] = scale_param*nrow(sub_df)/N_df
                Y[g,b] = scale_param*sum(sub_df[:,"y"])/N_df
                S[g,b] = mean(sub_df[:,"s"])
            end
        end
    end
    
    # Find totals
    Y_tot = sum(Y, dims=2);
    N_tot = sum(N, dims=2);
    
    return S, N, Y, N_tot, Y_tot
end

function get_fairness(x, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, print_output = true)
    # Compute the fairness gap in each bin then add up the absolute violations
    dp_viol = zeros(B)
    tpr_viol = zeros(B)
    fpr_viol = zeros(B)
    prp_viol = zeros(B)
    # Compute fairness violations
    for nb = 1:B
        dp_viol[nb] = (1/(N_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]) for ob in 1:B) - (1/(N_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]) for ob in 1:B)
        tpr_viol[nb] = (1/Y_tot[1])*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B) - (1/Y_tot[2])*sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B)
        fpr_viol[nb] = (1/(N_tot[1]-Y_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) - (1/(N_tot[2]-Y_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B)
        prp_viol[nb] = (sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B)/sum(x[2,ob,nb]*N[2,ob] for ob in 1:B)) - (sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B)/sum(x[1,ob,nb]*N[1,ob] for ob in 1:B))
    end
    # Compute AUC
    AUC = 0
    for g=1:G
        groupAUC = 0
        for k=1:B
            fprAtK = (1/(N_tot[g]-Y_tot[g]))*sum(x[g,ob,k]*(N[g,ob]-Y[g,ob]) for ob in 1:B)
            tprAtK = sum((1/Y_tot[g])*sum(x[g,ob,j]*Y[g,ob] for ob in 1:B) for j in 1:B if j >= k)
            groupAUC += fprAtK*tprAtK
        end
        AUC += (N_tot[g]/sum(N_tot))*groupAUC
    end
    if print_output
        println("AUC: ", AUC)
        println("Total DP violation: ", sum(abs.(dp_viol)))
        println("Worst DP violation: ", maximum(abs.(dp_viol)))
        println()
        println("Total TPR violation: ", sum(abs.(tpr_viol)))
        println("Worst TPR violation: ", maximum(abs.(tpr_viol)))
        println()
        println("Total FPR violation: ", sum(abs.(fpr_viol)))
        println("Worst FPR violation: ", maximum(abs.(fpr_viol)))
        println()
        println("Total PRP violation: ", sum(abs.(prp_viol)))
        println("Worst PRP violation: ", maximum(abs.(prp_viol)))
    end
    
    # Compute total score movement 
    score_movement = 0
    # println(G, B)
    for g=1:G
        for ob=1:B
            for nb=1:B
                # println(g, ob, nb)
                # println("error 1 ",bin_midpoints[nb])
                # println("error 2 ",S[g,ob])
                # println("error 3 ",N[g,ob])
                # println("error 4 ", x[g,ob,nb])
                score_movement += abs.(bin_midpoints[nb]-S[g,ob])*N[g,ob]*x[g,ob,nb]
            end
        end
    end
    return [score_movement, AUC, maximum(abs.(dp_viol)), maximum(vcat(abs.(tpr_viol), abs.(fpr_viol))), maximum(abs.(prp_viol))]
    
end

function get_solver_params(m)
    gap = relative_gap(m);
    objVal = objective_value(m);
    objBound = objective_bound(m);
    return gap, objVal, objBound
end

function solve_multifairness(G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, eps_dp, eps_eodds, eps_prp, max_movement, window_size, time_limit)
    """
    Solve_multifairness uses IpOpt to solve the nonconvex QCQP method through a barrier-based interior point optimization scheme.
    """
    m = Model()
    
    set_optimizer(m, Ipopt.Optimizer)
    set_optimizer_attribute(m, "max_cpu_time", time_limit)
    set_optimizer_attribute(m, "constr_viol_tol", global_constraint_tol)
    set_optimizer_attribute(m, "print_level", 0)
    
    @variable(m, 0 <= x[1:G, 1:B, 1:B] <= 1); # our fractional variables  
    if window_size > 0
        @constraint(m, [g=1:G, nb=1:B, ob=1:B;(ob < max(nb-window_size,1)) | (ob > min(nb+window_size,B))], x[g,ob,nb]==0);
    end
    
    @variable(m, 0 <= t_epi[1:G, 1:B, 1:B] <= 1); # epigraph variable for each bin difference for each group
    @constraint(m, [g=1:G, ob=1:B, nb=1:B], t_epi[g,ob,nb] >= abs.(bin_midpoints[nb]-S[g,ob])*N[g,ob]*x[g,ob,nb])
    @constraint(m, [g=1:G,b=1:B], sum(x[g,b,:]) == 1);
    @constraint(m, [g=1:G, nb=1:B,ob=1:B;nb==ob], x[g,ob,nb] >= 1-max_movement);
    
    # predictive rate parity constraints
    @constraint(m, [nb=1:B], eps_prp*sum(x[2,ob,nb]*N[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*N[1,ob] for ob in 1:B) >= 
    sum(x[2,ob,nb]*N[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B)-
    sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*N[1,ob] for ob in 1:B));
    @constraint(m, [nb=1:B], eps_prp*sum(x[2,ob,nb]*N[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*N[1,ob] for ob in 1:B) >= 
    sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*N[1,ob] for ob in 1:B)-
    sum(x[2,ob,nb]*N[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B));
    
    # rank order preservation constraints
    @constraint(m, [g=1:2, nb=1:(B-1)], sum(x[g,ob,nb+1]*Y[g,ob] for ob in 1:B)*sum(x[g,ob,nb]*N[g,ob] for ob in 1:B) 
        >= sum(x[g,ob,nb+1]*N[g,ob] for ob in 1:B)*sum(x[g,ob,nb]*Y[g,ob] for ob in 1:B))
    
    @constraint(m, [nb=1:(B-1)], eps_prp*sum(x[2,ob,nb]*N[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*N[1,ob] for ob in 1:B) >= 
    sum(x[2,ob,nb]*N[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B)-
    sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B)*sum(x[1,ob,nb]*N[1,ob] for ob in 1:B));
    
    # Equal TPR constraints
    @constraint(m, [nb=1:B], (1/Y_tot[1])*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B) - (1/Y_tot[2])*sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B) <= eps_eodds);
    @constraint(m, [nb=1:B], (1/Y_tot[2])*sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B) - (1/Y_tot[1])*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B) <= eps_eodds);
    # Equal TNR constraints 
    @constraint(m, [nb=1:B], (1/(N_tot[1]-Y_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) - (1/(N_tot[2]-Y_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) <= eps_eodds);
    @constraint(m, [nb=1:B], (1/(N_tot[2]-Y_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) - (1/(N_tot[1]-Y_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) <= eps_eodds);
    
    # DP
    @constraint(m, [nb=1:B], (1/(N_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]) for ob in 1:B) - 
    (1/(N_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]) for ob in 1:B) <= eps_dp);
    @constraint(m, [nb=1:B], (1/(N_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]) for ob in 1:B) -
    (1/(N_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]) for ob in 1:B) <= eps_dp);
    
    # Finally, set up the to minimize the score movement
    @objective(m, Min, sum(t_epi));
    optimize!(m)
    
    return value.(x), m
end

function get_den_bounds(G, B, S, N, Y, N_tot, Y_tot, arg_g, arg_b, max_movement, window_size, eps_dp, eps_eodds, obj)
    m = Model()
    set_optimizer(m, Gurobi.Optimizer)
    set_optimizer_attribute(m, "OutputFlag", 0)
    
    @variable(m, 0 <= x[1:G, 1:B, 1:B] <= 1);
    if window_size > 0
        @constraint(m, [g=1:G, nb=1:B, ob=1:B;(ob < max(nb-window_size,1)) | (ob > min(nb+window_size,B))], x[g,ob,nb]==0);
    end
    @constraint(m, [g=1:G,b=1:B], sum(x[g,b,:]) == 1);
    @constraint(m, [g=1:G, nb=1:B,ob=1:B;nb==ob], x[g,ob,nb] >= 1-max_movement);
    # Equal TPR constraints
    @constraint(m, [nb=1:B], (1/Y_tot[1])*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B) - (1/Y_tot[2])*sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B) <= eps_eodds);
    @constraint(m, [nb=1:B], (1/Y_tot[2])*sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B) - (1/Y_tot[1])*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B) <= eps_eodds);
    # Equal FNR constraints 
    @constraint(m, [nb=1:B], (1/(N_tot[1]-Y_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) - (1/(N_tot[2]-Y_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) <= eps_eodds);
    @constraint(m, [nb=1:B], (1/(N_tot[2]-Y_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) - (1/(N_tot[1]-Y_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) <= eps_eodds);
    # DP constraints 
    @constraint(m, [nb=1:B], (1/(N_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]) for ob in 1:B) - 
    (1/(N_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]) for ob in 1:B) <= eps_dp);
    @constraint(m, [nb=1:B], (1/(N_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]) for ob in 1:B) -
    (1/(N_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]) for ob in 1:B) <= eps_dp);
    
    if obj == "min"
        @objective(m, Min, sum(x[arg_g,ob,arg_b]*N[arg_g,ob] for ob in 1:B));
    else
        @objective(m, Max, sum(x[arg_g,ob,arg_b]*N[arg_g,ob] for ob in 1:B));
    end
    
    optimize!(m);
    return objective_value(m)
end


function get_t_bounds(G, B, S, N, Y, N_tot, Y_tot, arg_g, arg_b, den_bounds, max_movement, window_size, eps_dp, eps_eodds, obj)
    m = Model()
    set_optimizer(m, Gurobi.Optimizer)
    set_optimizer_attribute(m, "OutputFlag", 0)
    
    @variable(m, 0 <= ξ[1:G, 1:B, 1:B] <= 1/den_bounds[arg_g,arg_b,1]);
    if window_size > 0
        @constraint(m, [g=1:G, nb=1:B, ob=1:B;(ob < max(nb-window_size,1)) | (ob > min(nb+window_size,B))], ξ[g,ob,nb]==0);
    end
    @variable(m, 1/den_bounds[arg_g,arg_b,2] <= ϕ <= 1/den_bounds[arg_g,arg_b,1]);
    
    @constraint(m, sum(ξ[arg_g, b, arg_b]*N[arg_g, b] for b in 1:B) == 1.0);
    @constraint(m, [g=1:G, nb=1:B,ob=1:B;nb==ob], ξ[g, ob, nb] >= (1-max_movement)*ϕ); 
    
    # Equal TPR 
    @constraint(m, [nb=1:B], (1/Y_tot[1])*sum(ξ[1,ob,nb]*Y[1,ob] for ob in 1:B) - (1/Y_tot[2])*sum(ξ[2,ob,nb]*Y[2,ob] for ob in 1:B) <= eps_eodds*ϕ);
    @constraint(m, [nb=1:B], (1/Y_tot[2])*sum(ξ[2,ob,nb]*Y[2,ob] for ob in 1:B) - (1/Y_tot[1])*sum(ξ[1,ob,nb]*Y[1,ob] for ob in 1:B) <= eps_eodds*ϕ);
    # Equal TNR 
    @constraint(m, [nb=1:B], (1/(N_tot[1]-Y_tot[1]))*sum(ξ[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) - (1/(N_tot[2]-Y_tot[2]))*sum(ξ[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) <= eps_eodds*ϕ);
    @constraint(m, [nb=1:B], (1/(N_tot[2]-Y_tot[2]))*sum(ξ[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) - (1/(N_tot[1]-Y_tot[1]))*sum(ξ[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) <= eps_eodds*ϕ);
    # DP
    @constraint(m, [nb=1:B], (1/(N_tot[1]))*sum(ξ[1,ob,nb]*(N[1,ob]) for ob in 1:B) - 
    (1/(N_tot[2]))*sum(ξ[2,ob,nb]*(N[2,ob]) for ob in 1:B) <= eps_dp*ϕ);
    @constraint(m, [nb=1:B], (1/(N_tot[2]))*sum(ξ[2,ob,nb]*(N[2,ob]) for ob in 1:B) -
    (1/(N_tot[1]))*sum(ξ[1,ob,nb]*(N[1,ob]) for ob in 1:B) <= eps_dp*ϕ);
    
    if obj == "min"
        @objective(m, Min, sum(ξ[arg_g,ob,arg_b]*Y[arg_g,ob] for ob in 1:B));
    else
        @objective(m, Max, sum(ξ[arg_g,ob,arg_b]*Y[arg_g,ob] for ob in 1:B));
    end
    
    optimize!(m);
    return objective_value(m)
end

function validate_t_bounds(x,G, B, t_bounds_re, N, Y)
    # Debugging function 
    for g in 1:G
        for nb in 1:B
            bin_exp = (sum(x[g,ob,nb]*Y[g,ob] for ob in 1:B)/sum(x[g,ob,nb]*N[g,ob] for ob in 1:B))
            @assert( (t_bounds_re[g,nb,1]-1e-4) <= bin_exp <= (t_bounds_re[g,nb,2]+1e-4))
        end
    end
end

function solve_multifairness_binary(G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, eps_dp, eps_eodds, eps_prp, max_movement, window_size, time_limit)
    """
    Solves the reformulated problem as a MILP using the proposed methodology. Makes calls to get_t_bounds and get_den_bounds to bound the bilinear terms first, then via binary expansion of bilinear product.  
    """
    den_bounds = zeros((G,B,2))
    t_bounds = zeros((G,B,2))

    for g=1:G
        for b=1:B
            den_bounds[g,b,1] = get_den_bounds(G, B, S, N, Y, N_tot, Y_tot, g, b, max_movement, window_size, eps_dp, eps_eodds, "min")
            den_bounds[g,b,2] = get_den_bounds(G, B, S, N, Y, N_tot, Y_tot, g, b, max_movement, window_size, eps_dp, eps_eodds, "max")
        end
    end

    for g=1:G
        for b=1:B
            t_bounds[g,b,1] = get_t_bounds(G, B, S, N, Y, N_tot, Y_tot, g, b, den_bounds, max_movement, window_size, eps_dp, eps_eodds, "min")
            t_bounds[g,b,2] = get_t_bounds(G, B, S, N, Y, N_tot, Y_tot, g, b, den_bounds, max_movement, window_size, eps_dp, eps_eodds, "max")
        end
    end
    
    m = Model(()->QuadraticToBinary.Optimizer{Float64}(
            MOI.instantiate(
                optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0,
                    "MIPFocus" => 1, "MIPGap" => 0.20, "FeasibilityTol" => global_constraint_tol), 
                with_bridge_type = Float64)))
    
    MOI.set(m, QuadraticToBinary.GlobalVariablePrecision(), global_constraint_tol)
    set_time_limit_sec(m, time_limit) 
    
    @variable(m, 0 <= x[1:G, 1:B, 1:B] <= 1); # our fractional variables  
    # Add window constraints
    if window_size > 0
        @constraint(m, [g=1:G, nb=1:B, ob=1:B;(ob < max(nb-window_size,1)) | (ob > min(nb+window_size,B))], x[g,ob,nb]==0);
    end
    @variable(m, 0 <= t_epi[1:G, 1:B, 1:B] <= 1); # epigraph variable for each bin difference for each group
    @constraint(m, [g=1:G, ob=1:B, nb=1:B], t_epi[g,ob,nb] >= abs.(bin_midpoints[nb]-S[g,ob])*N[g,ob]*x[g,ob,nb])
    @constraint(m, [g=1:G,b=1:B], sum(x[g,b,:]) == 1);
    @constraint(m, [g=1:G, nb=1:B,ob=1:B;nb==ob], x[g,ob,nb] >= 1-max_movement);
    
    # prp constraints with transformation and bounds
    @variable(m, t_bounds[g,b,1] <= t[g=1:G, b=1:B] <= t_bounds[g,b,2]);
    @variable(m, den_bounds[g,b,1] <= z[g=1:G, b=1:B] <= den_bounds[g,b,2]); 
    @constraint(m, [g=1:2, nb=1:B], sum(x[g,ob,nb]*Y[g,ob] for ob in 1:B) == z[g,nb]*t[g,nb])
    
    @constraint(m, [g=1:2, nb=1:B], sum(x[g,ob,nb]*N[g,ob] for ob in 1:B) == z[g,nb]);
    @constraint(m, [nb=1:B], t[1,nb] - t[2,nb] <= eps_prp);
    @constraint(m, [nb=1:B], t[2,nb] - t[1,nb] <= eps_prp);
    
    # rank order preservation constraints
    @constraint(m, [g=1:2, nb=1:(B-1)], t[g,nb+1] >= t[g,nb])
    
    # Equal TPR constraints
    @constraint(m, [nb=1:B], (1/Y_tot[1])*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B) - (1/Y_tot[2])*sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B) <= eps_eodds);
    @constraint(m, [nb=1:B], (1/Y_tot[2])*sum(x[2,ob,nb]*Y[2,ob] for ob in 1:B) - (1/Y_tot[1])*sum(x[1,ob,nb]*Y[1,ob] for ob in 1:B) <= eps_eodds);
    # Equal TNR constraints 
    @constraint(m, [nb=1:B], (1/(N_tot[1]-Y_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) - (1/(N_tot[2]-Y_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) <= eps_eodds);
    @constraint(m, [nb=1:B], (1/(N_tot[2]-Y_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]-Y[2,ob]) for ob in 1:B) - (1/(N_tot[1]-Y_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]-Y[1,ob]) for ob in 1:B) <= eps_eodds);
    
    # DP
    @constraint(m, [nb=1:B], (1/(N_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]) for ob in 1:B) - 
    (1/(N_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]) for ob in 1:B) <= eps_dp);
    @constraint(m, [nb=1:B], (1/(N_tot[2]))*sum(x[2,ob,nb]*(N[2,ob]) for ob in 1:B) -
    (1/(N_tot[1]))*sum(x[1,ob,nb]*(N[1,ob]) for ob in 1:B) <= eps_dp);
    
    # Finally, set up the to minimize the score movement
    @objective(m, Min, sum(t_epi));
    optimize!(m)
    
    return value.(x), m
end

function apply_random_transitions(df, x, seed)
    dfr=df;
    dfr[:,"newbin"] .= 0;
    # create y_predictions with n rows = df's rows
    # y_predictions = zeros(nrow(df))

    # Want to check that our mapping indeed reduces the OT difference below our desired ϵ in expectation
    for b=1:B
        for g=1:G
            df_b = df[(df[:,"bin"].==b) .& (df[:,"g"].==g) ,:]
            n_samples = nrow(df_b)
            # now apply the mapping that we found above to the testing set by looking at the row-wise transformation 
            probabilities = x[g,b,:]
            items = [i for i in 1:length(probabilities)]
            weights = Weights(probabilities)
            Random.seed!((seed+1)*b*g)
            mappings = [sample(items,weights) for x=1:n_samples]
            # apply mappings
            df[(df[:,"bin"].==b) .& (df[:,"g"].==g), "newbin"] .= mappings 
        end
    end
    # create y_predict variable to store the boolean comparison of the new bin and the old bin 
    y_predictions = dfr[!, :newbin]
    dfr[!, :bin] = dfr[!, :newbin];
    
    return dfr[:, [:s, :y, :g, :bin]], y_predictions
end

# Define common problem parameters
G = 2
B = 2
n_trials = 1
max_movement = 0.5
reduction = 0.5
window_size = Int(round(B/2))
time_limit = 60.0*10.0

# results_method = zeros((n_trials, 5));
results_base = zeros((n_trials, 5));
results_opt = zeros((n_trials, 5));

# results_method_test = zeros((n_trials, 5));
results_base_test = zeros((n_trials, 5));
results_opt_test = zeros((n_trials, 5));

"""
Define task name and path. Expect the path to contain 
- "{task}_train.csv", "{task}_bin_midpoints.csv", "{task}_test.csv" which are the training and testing data from the original (base) model
- "{task}_train_{method}.csv", "{task}_test_{method}.csv"  which are the training and testing data from the regularized model (Pleiss Eodds or Razaei FairLogLoss) model
Require that train and test file (for both cases) contains 4 columns --> | score [0,1] | label {0,1} | group {1,2} | bin {1,2,...,B}|
Require that midpoints file contains a single column ("bin_midpoints") with B scalar values representing the bin midpoints (midpoint between quantiles)
"""
println(ARGS)
data_path = ARGS[1] # path to data with bin_midpoints, train, test
task_name = ARGS[2]
# data_path = "/home/lvu5/temp-fair/MFOpt/Data Generation/MultipleFairness/Data/acs_west_income/" # Path to 
# task_name = "acs_west_income"
# Name of the task
# method = "fll" # "fll" or "pleiss"

y_predictions = []

for i in 1:n_trials
    folder = "Trial_"*string(i-1)*"/"
    path = data_path*"/"*task_name
    # Get base fairness results
    x_eye = zeros((G,B,B))
    x_eye[1,:,:] = I(B)
    x_eye[2,:,:] = I(B)

    # Get metrics for training data
    # df = DataFrame(CSV.File(path*"_train_"*method*".csv"));
    # midpoints = DataFrame(CSV.File(path*"_bin_midpoints.csv"));
    df = DataFrame(CSV.File(path*"_train.csv"));
    midpoints = DataFrame(CSV.File(path*"_bin_midpoints.csv"));
    bin_midpoints = midpoints[!, "bin_midpoints"];
    # bin_midpoints = midpoints["bin_midpoints"];
    # S, N, Y, N_tot, Y_tot = get_parameters(df, G, B, true, 100);
    # scoreObj, auc_method, dpViol_method, eoddsViol_method, prpViol_method = get_fairness(x_eye,G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    # results_method[i,:] = [scoreObj, auc_method, dpViol_method, eoddsViol_method, prpViol_method];

    df = DataFrame(CSV.File(path*"_train.csv"));
    S, N, Y, N_tot, Y_tot = get_parameters(df, G, B, true, 100);
    scoreObj, auc_base, dpViol_base, eoddsViol_base, prpViol_base = get_fairness(x_eye, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    results_base[i,:] = [scoreObj, auc_base, dpViol_base, eoddsViol_base, prpViol_base];

    eps_dp = min(1, dpViol_base)*reduction
    eps_eodds = min(1, eoddsViol_base)*reduction
    eps_prp = min(1, prpViol_base)*reduction
    
    x_opt_bin, m_bin = solve_multifairness(G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, eps_dp, eps_eodds, eps_prp, max_movement, window_size, time_limit);
    scoreObj, auc_opt, dpViol_opt, eoddsViol_opt, prpViol_opt = get_fairness(x_opt_bin, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    results_opt[i,:] = [scoreObj, auc_opt, dpViol_opt, eoddsViol_opt, prpViol_opt];

    # Apply on testing data
    # df = DataFrame(CSV.File(path*"_test_"*method*".csv"));
    # S, N, Y, N_tot, Y_tot = get_parameters(df, G, B, true, 100);
    # scoreObj, auc_method, dpViol_method, eoddsViol_method, prpViol_method = get_fairness(x_eye, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    # results_method_test[i,:] = [scoreObj, auc_method, dpViol_method, eoddsViol_method, prpViol_method];

    df = DataFrame(CSV.File(path*"_test.csv"));
    S, N, Y, N_tot, Y_tot = get_parameters(df, G, B, true, 100);
    scoreObj, auc_base, dpViol_base, eoddsViol_base, prpViol_base = get_fairness(x_eye, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    results_base_test[i,:] = [scoreObj, auc_base, dpViol_base, eoddsViol_base, prpViol_base];

    df = DataFrame(CSV.File(path*"_test.csv"));
    dfr, y_predictions = apply_random_transitions(df, x_opt_bin, i*5);
    # println("y_predictions = ", y_predictions)
    # write dfr to csv
    save_path = "./experiments/MFOpt_results/"
    CSV.write(save_path*task_name*"_test_opt_label.csv", dfr)
    S, N, Y, N_tot, Y_tot = get_parameters(dfr, G, B, true, 100);
    scoreObj, auc_opt, dpViol_opt, eoddsViol_opt, prpViol_opt = get_fairness(x_eye, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, true)
    results_opt_test[i,:] = [scoreObj, auc_opt, dpViol_opt, eoddsViol_opt, prpViol_opt];
end

# Save results
# results_method_df = DataFrame(results_method, [:scoreObj, :auc, :dpViol, :eoddsViol, :prpViol])
# CSV.write(data_path*"train_method.csv", results_method_df)
# println("check ehereererererer")
println(data_path)
save_path = "./experiments/MFOpt_results/"

results_base_df = DataFrame(results_base, [:scoreObj, :auc, :dpViol, :eoddsViol, :prpViol])
CSV.write(save_path*task_name*"_train_base.csv", results_base_df)
results_opt_df = DataFrame(results_opt, [:scoreObj, :auc, :dpViol, :eoddsViol, :prpViol])
CSV.write(save_path*task_name*"_train_opt.csv", results_opt_df)

# results_method_df = DataFrame(results_method_test, [:scoreObj, :auc, :dpViol, :eoddsViol, :prpViol])
# CSV.write(data_path*"test_method.csv", results_method_df)
results_base_df = DataFrame(results_base_test, [:scoreObj, :auc, :dpViol, :eoddsViol, :prpViol])
CSV.write(save_path*task_name*"_test_base.csv", results_base_df)
results_opt_df = DataFrame(results_opt_test, [:scoreObj, :auc, :dpViol, :eoddsViol, :prpViol])
CSV.write(save_path*task_name*"_test_opt.csv", results_opt_df)

# println(y_predictions)

