        # add camera tensors
        if self.cfg.cam.camera:
            """             
            self.projection_matrix = []
            self.view_matrix = []
            for i in range(self.num_envs):
                self.projection_matrix.append(np.matrix(self.gym.get_camera_proj_matrix(self.sim, self.envs[i], self.camera_handles[i])))
                self.view_matrix.append(np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i]))) """

            
            zeros = torch.zeros(self.num_envs, self.cfg.cam.num_obs_cam)
            zeros = zeros.to(self.device)
            self.obs_buf = torch.cat((self.obs_buf, zeros), dim=-1)
            
        
            self.base_type = networks.MLPBase
            # params['net']['activation_func'] = torch.nn.Tanh
            self.in_channels = 1

            encoder = networks.LocoTransformerEncoder(
                
                in_channels=self.in_channels, #env.image_channels,
                state_input_dim=self.obs_buf.shape[0],
                hidden_shapes=(200,200),
            )

            # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

            if torch.cuda.is_available():
                encoder.cuda()
            #     encoder.to(device)
            
            for i in range(self.num_envs):
                img = self.camera_tensors[i].reshape(1,1,self.cfg.cam.width,self.cfg.cam.height)
                img[img == float("-Inf")] = -3
                print('img', img)
                # print('ten', self.camera_tensors[i])
                obs = self.obs_buf[i,:235]     
                # print('obs', obs.get_device())       
                # print('img',img)
                # obs = obs.to(device)
                # img = img.to(device)
                with torch.no_grad(): 
                    visual_out, state_out = encoder(img, obs)
                    # state_out = pf()
                    # visual_out = vf()
                # print('vo', visual_out.shape, 'so', state_out.shape)
                # Normalization
                # visual_out = networks.NormObsWithImg(visual_out)     
                self.obs_buf[i] = visual_out
                # print('f',i, self.obs_buf[:,235:])    
            
            

            # self.obs_buf.to(device)
            # print('obs2', self.obs_buf.get_device())
            # print('f',self.obs_buf[:,:20])

            # print('f',self.obs_buf[:,235:])
             
            # print('f',type(self.features), self.features.shape)