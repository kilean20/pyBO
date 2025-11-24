                            
    def cutomzie_acqu(self,
                      acquisition_func=None,
                      acquisition_func_args=None,
                      UCB_beta = None,
                      best_y = None,
                      acquisition_optimize_options = None,
                      X_pending = None,
                      Y_pending_future = None,
                      X_penal = None, 
                      L_penal = None,
                      C_penal = None,
                      X_favor = None,
                      L_favor = None,
                      C_favor = None,
                      polarity_penalty = None,
                      ramping_rate = None,
                      polarity_change_time = None,
                      ):
               
        if acquisition_func is None:
            acquisition_func = self.acquisition_func
                
        if acquisition_func_args is None:
            acquisition_func_args = {}
        else:
            acquisition_func_args = copy(acquisition_func_args)
        if acquisition_func.name == 'ExpectedImprovement':
            if best_y is None:
                if self.y is not None:
                    best_y = np.max(self.y)
                else:
                    raise ValueError("'best_y' is required for EI")
            acquisition_func_args['best_y'] = best_y
            
        if acquisition_func.name == 'UpperConfidenceBound':
            if UCB_beta is not None:
                beta = UCB_beta
            elif 'beta' in acquisition_func_args:
                beta = acquisition_func_args['beta']
            else:
                beta = 9.0
            acquisition_func_args['beta'] = beta   
              
        if X_pending is not None:
            X_pending = np.atleast_2d(X_pending)
        x1 = np.zeros((batch_size,self.ndim),dtype=dtype)
        for q in range(batch_size):   
            def acqu(x):
                return acquisition_func(x,**acquisition_func_args)
            if X_penal is None:
                X_penal = self.x
            penal_favor_acqu_args = self._auto_penal_favor(X_penal,L_penal,C_penal,
                                                           X_favor,L_favor,C_favor,
                                                           X_pending,polarity_penalty,
                                                           bounds = bounds,
                                                           acquisition=acqu)
            if debug:
                print("--in pyBO acqu query--")
                print("  [debug]acquisition_func_args",acquisition_func_args)
                print("  [debug]penal_favor_acqu_args",penal_favor_acqu_args)
                
                
            for key, val in penal_favor_acqu_args.items():
                acquisition_func_args[key] = val
                
            def acqu(x):
                return acquisition_func(x,**acquisition_func_args)

            
          
            acqu_bounds = np.array(bounds)
            if fixed_values_for_each_dim is not None:
                def acqu(x,fixed_values_for_each_dim=fixed_values_for_each_dim):
                    return float(acquisition_func(self._insert_fixed_values(x,fixed_values_for_each_dim),
                                                  **acquisition_func_args))
                acqu_bounds = []
                for i,b in enumerate(bounds):
                    if i not in fixed_values_for_each_dim.keys():
                        acqu_bounds.append(b)
                acqu_bounds = np.array(acqu_bounds)
                
            if optmization_stopping_criteria is None:
                optmization_stopping_criteria = []
                if batch_size>1:
                    if timeout is None:
                        warn("'timeout' input is recommend for multi-batch candidate querry. By default timeout = 5 sec for each candidate search")
                        timeout = 5*batch_size  
                    optmization_stopping_criteria.append(util.timeout(timeout))
                if Y_pending_future is not None:
                    optmization_stopping_criteria.append(Y_pending_future.done)
                if len(optmization_stopping_criteria) == 0:
                    optmization_stopping_criteria = None

            acquisition_optimize_options['optmization_stopping_criteria'] = optmization_stopping_criteria
            if X_pending is not None:
                if X_pending.ndim != 2 or len(X_pending)<1:
                    print("in acqu query X_pending",X_pending)
                acquisition_optimize_options['x0'] = np.clip(X_pending[-1,:] + 0.02*np.random.randn()*(acqu_bounds[:,1]-acqu_bounds[:,0]),
                                                             a_min=acqu_bounds[:,0],
                                                             a_max=acqu_bounds[:,1],)

            result = util.maximize_with_restarts(acqu,bounds=acqu_bounds,
                                                 **acquisition_optimize_options,
                                                 **scipy_minimize_options)

            acqu_bounds_diff = acqu_bounds[:,1]-acqu_bounds[:,0]
            if not hasattr(result,"x"):
                if X_pending is not None:
                    warn('Scipy minimize could not find new candidate. Old candidate will be used instead')
                    result.x = X_pending[-1,:]  + 0.2*np.random.randn()*acqu_bounds_diff
                else:
                    warn('Scipy minimize could not find new candidate. Random candidate will be used instead')
                    result.x = np.random.rand(len(acqu_bounds))*acqu_bounds_diff +acqu_bounds[:,0] 

            # else:
            #     bounds_diff = (self.bounds[:,1] - self.bounds[:,0]).reshape(1,-1)
            #     if  dist_neighbor <= dist10percent:
            #         if X_pending is not None:
            #             result.x = X_pending[-1,:]  + 0.2*np.random.randn()*acqu_bounds_diff
            #             result.x = np.clip(result.x, a_min = acqu_bounds[:,0], a_max = acqu_bounds[:,1])
            #             while np.all(result.x == X_pending[-1,:]):
            #                 result.x = X_pending[-1,:]  + 0.2*np.random.randn()*acqu_bounds_diff
            #                 result.x = np.clip(result.x, a_min = acqu_bounds[:,0], a_max = acqu_bounds[:,1])
            #         else:
            #             result.x = np.random.rand(len(acqu_bounds))*0.5*acqu_bounds_diff +0.5*(acqu_bounds[:,0]+acqu_bounds[:,1])
            result.x = np.clip(result.x, a_min = acqu_bounds[:,0], a_max = acqu_bounds[:,1])
            bounds_diff = (self.bounds[:,1] - self.bounds[:,0]).reshape(1,-1)
            i_try = 1
            while np.mean(np.abs(result.x - X_pending[-1,:])/bounds_diff) < 1e-6 and i_try < 10:
                result.x = X_pending[-1,:]  + 0.2*np.random.randn()*acqu_bounds_diff
                result.x = np.clip(result.x, a_min = acqu_bounds[:,0], a_max = acqu_bounds[:,1])
                i_try += 1
            
#                 print("=== debug ===")
#                 print("result",result)
#                 print("acquisition_optimize_options",acquisition_optimize_options)
#                 print("scipy_minimize_options",scipy_minimize_options)
#                 print("acqu_bounds",acqu_bounds)
#                 if 'x0' in acquisition_optimize_options:
#                     print("acqu(acquisition_optimize_options['x0'])",acqu(acquisition_optimize_options['x0']))
            if fixed_values_for_each_dim is None:
                x1[q,:] = result.x.flatten()
            else:
                x1[q,:] = self._insert_fixed_values(result.x,fixed_values_for_each_dim).flatten()
            
            # polarity_penalty does not keep track of past candidate.
            # strict zero makes polarity_penalty mis-behave
            is_zero = x1[q,:]==0
            if X_pending is not None:
                x1[q,is_zero] = 1e-6*np.sign(X_pending[-1,is_zero])
                
            hist = self.history[-1] 
            hist['acquisition'].append(copy(acquisition_func))
            hist['bounds'].append(copy(bounds))
            hist['acquisition_args'].append(copy(acquisition_func_args))
            hist['x1'].append(copy(x1[q:q+1,:]))
            if write_log:
                self.write_log(path=self.path, tag=self.tag)
            hist['query_time'].append(time.monotonic()-start_time)
            
            
            if X_pending is None:
                X_pending = x1[:q+1,:]
            else:
                X_pending = np.vstack((X_pending,x1[q:q+1,:]))
                
#         return x1
#         fig,axs = plt.subplots(1,2,figsize=(8,3))
#         ax=axs[0]
#         ax.plot(x1[:,0],x1[:,1],alpha=0.5,c='r')
#         print("before x1.shape",x1.shape)
#         time.sleep(1)
#         ax.scatter(x1[:,0],x1[:,1],s=np.linspace(4,16,len(x1)))
#         ax.hlines(0,bounds[0,0],bounds[0,1],ls='--',color='k')
#         ax.vlines(0,bounds[1,0],bounds[1,1],ls='--',color='k')
#         ax.scatter(X_pending[:1,0],X_pending[:1,1],c='k')
#         ax.set_xlim(bounds[0,0],bounds[0,1])
#         ax.set_ylim(bounds[1,0],bounds[1,1])
#         ax=axs[1]
#         x1 = util.order_samples(x1,
#                                x0=X_pending[:1,:],
#                                ramping_rate=ramping_rate,
#                                polarity_change_time=polarity_change_time)
#         print("after x1.shape",x1.shape)
#         time.sleep(1)
#         ax.plot(x1[:,0],x1[:,1],alpha=0.5,c='r')
#         ax.scatter(x1[:,0],x1[:,1],s=np.linspace(4,16,len(x1)))
#         ax.hlines(0,bounds[0,0],bounds[0,1],ls='--',color='k')
#         ax.vlines(0,bounds[1,0],bounds[1,1],ls='--',color='k')
#         ax.scatter(X_pending[:1,0],X_pending[:1,1],c='k')
#         ax.set_xlim(bounds[0,0],bounds[0,1])
#         ax.set_ylim(bounds[1,0],bounds[1,1])
        x1 = np.unique(x1, axis=0)

        return  util.order_samples(x1,
                                   x0=X_pending[:1,:],
                                   ramping_rate=ramping_rate,
                                   polarity_change_time=polarity_change_time)
