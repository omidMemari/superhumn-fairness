        self.model_obj.optimizer.zero_grad()
        # loss = multiply(prob(y_hat given Xi)) * subdom_tensor
        self.subdom_tensor = np.zeros((self.num_of_demos, self.num_of_features)) 
        self.sample_loss = np.zeros((self.num_of_demos, self.num_of_features))
        loss = 0
        for j, demo in enumerate(tqdm(self.demo_list)):
          # breakpoint()
          X_test = X[demo.idx_test.to_numpy(),:]
          pred = self.model_obj(X_test)
          y_hat_probs, _ = torch.max(pred, dim=1)
          with torch.no_grad():
            y_hat = torch.argmax(pred,dim=1).cpu().detach().numpy()
          models_dict = {"Super_human": (y_hat, y_hat) }
          y = self.train_data.loc[demo.idx_test][self.label] # we use true_y
          A = self.train_data.loc[demo.idx_test][self.sensitive_feature]
          A_str = A.map(self.dict_map)
          try:
            metric_df = get_metrics_df(models_dict = models_dict, y_true = y, group = A_str,\
            feature = self. feature, is_demo = False)
          except:
            import ipdb; ipdb.set_trace()
          f_hat = []
          f_tilde = []
          #f_demo vector
          for feature_index in range(self.num_of_features):
            f_hat.append(metric_df.loc[self.feature[feature_index]]["Super_human"])
            f_tilde.append(demo.metric[feature_index])
          f_hat = np. asarray(f_hat)
          self.sample_loss [j, :] = f_hat # update sample loss matrix
          f_tilde = np.asarray(f_tilde)
          subdom = np.maximum(np.zeros(self.num_of_features), alpha*(f_hat - f_tilde) + beta).sum()
          # (1, )
          log_prob_sum = torch.log(y_hat_probs).sum()
          loss += log_prob_sum * subdom