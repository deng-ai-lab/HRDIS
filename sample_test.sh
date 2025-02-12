samples_root=./_exp
save_deg=True
save_ori=True
overwrite=True
smoke_test=1000
batch_size=4
num_steps=100



#Nonlinear HDR
#HDR + reddiff + adam
#CUDA_VISIBLE_DEVICES=1 python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=hdr    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    dist.num_processes_per_node=1   exp.name=reddiffHDR  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25   exp.save_evolution=True   algo.lr=0.5  exp.start_step=1000   exp.end_step=0

#HDR + HRDIS + adam
#CUDA_VISIBLE_DEVICES=2 python   main.py   exp.overwrite=True   algo=HRDIS   algo.awd=True    algo.deg=hdr    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    dist.num_processes_per_node=1   exp.name=HRDISHDR  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25   exp.save_evolution=True   algo.lr=0.1  exp.start_step=1000   exp.end_step=0

#HDR + dps
#CUDA_VISIBLE_DEVICES=3 python   main.py   exp.overwrite=True   algo=dps   algo.awd=True    algo.deg=hdr    algo.eta=0.5    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3     dist.num_processes_per_node=1   exp.name=dpsHDR2  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.05



#Nonlinear deblurring
#deblur + reddiff + adam
#CUDA_VISIBLE_DEVICES=2 python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=deblur_nl    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=redddiff_nldeblur2  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.25 algo.obs_weight=0.02 algo.denoise_term_weight=const exp.save_evolution=False   algo.lr=0.25  exp.start_step=1000   exp.end_step=100

#deblur + HRDIS + adam
#CUDA_VISIBLE_DEVICES=2 python   main.py   exp.overwrite=True   algo=HRDIS   algo.awd=True    algo.deg=deblur_nl    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=HRDIS_nldeblur_beta0.5  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root   exp.save_evolution=True   algo.lr=0.1  algo.obs_weight=0.02 algo.denoise_term_weight=const exp.start_step=1000   exp.end_step=100

#deblur + dps
#CUDA_VISIBLE_DEVICES=2 python   main.py   exp.overwrite=True   algo=dps   algo.awd=True    algo.deg=deblur_nl    algo.eta=0.5    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=dps_nldeblur  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root  algo.grad_term_weight=0.05



#SR
#sr + reddiff + adam
#UDA_VISIBLE_DEVICES=7 python   main.py   exp.overwrite=True   algo=reddiff   algo.awd=True    algo.deg=sr16    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=reddsr4_ffhq  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root algo.lr=0.5

#sr + HRDIS + adam
#CUDA_VISIBLE_DEVICES=2 python   main.py   exp.overwrite=True   algo=HRDIS   algo.awd=True    algo.deg=sr16    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=6    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=HRDISsr4_ffhq_test  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root algo.lr=0.5 algo.const=0.2

#sr + pgdm 
#CUDA_VISIBLE_DEVICES=1 python   main.py   exp.overwrite=True   algo=pgdm   algo.awd=True    algo.deg=sr16    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=5    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=pisr4_ffhq  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test  exp.samples_root=$samples_root

#sr + dps
#choose eta
#grad_term_weight=0.1 #0.25 0.5 1.0 2.0
#CUDA_VISIBLE_DEVICES=7 python   main.py   exp.overwrite=$overwrite   algo=dps    algo.deg=sr16    algo.eta=0.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=dpssr_ffhq  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root   algo.grad_term_weight=1.0

#sr + ddrm
#CUDA_VISIBLE_DEVICES=7 python   main.py   exp.overwrite=True   algo=ddrm    algo.deg=sr16    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=ddrmsr_ffhq  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root #-cn imagenet256_cond   #model.ckpt=imagenet256_cond

 

       

#INPAINT
#inpaint + reddiff + adam
#CUDA_VISIBLE_DEVICES=7 python   main.py   exp.overwrite=True   algo=reddiff  exp.seed=1  algo.sigma_x0=0.0001   algo.lr=0.1   algo.awd=True    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.1   loader.batch_size=$batch_size    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=reddiff_inp_sigma0.1_imagenet  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root  exp.end_step=100

#inpaint + HRDIS + adam
CUDA_VISIBLE_DEVICES=1 python   main.py   dist.num_processes_per_node=1 exp.overwrite=True   algo=HRDIS  exp.seed=5  algo.sigma_x0=0.0001   algo.lr=0.1   algo.awd=True    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.05   loader.batch_size=$batch_size    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=HRDIS_inp_sigma0.05_imagenet_2  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root 

#inpaint + pgdm
#CUDA_VISIBLE_DEVICES=6 python   main.py   exp.overwrite=True   algo=pgdm   algo.awd=True    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=150    algo.sigma_y=0.1   loader.batch_size=$batch_size    exp.seed=5    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=pgdm_inp_sigma0.1_imagenet  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root #-cn imagenet256_cond   #model.ckpt=imagenet256_cond

#inpaint + dps
#choose eta
#grad_term_weight 0.1 0.25 0.5 1.0 2.0
#CUDA_VISIBLE_DEVICES=1 python   main.py   exp.overwrite=True   algo=dps    algo.deg=in2_20ff    algo.eta=0.0    exp.num_steps=$num_steps    algo.sigma_y=0.0   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=test_time  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root   algo.grad_term_weight=0.25

#inpaint + ddrm
#CUDA_VISIBLE_DEVICES=0 python   main.py   exp.overwrite=True   algo=ddrm    algo.deg=in2_20ff    algo.eta=1.0    exp.num_steps=$num_steps    algo.sigma_y=0.1   loader.batch_size=$batch_size    exp.seed=3    loader=imagenet256_ddrmpp    dist.num_processes_per_node=1   exp.name=ddrm_inp_sigma0.1_imagenet  exp.save_ori=$save_ori  exp.save_deg=$save_deg  exp.smoke_test=$smoke_test   exp.samples_root=$samples_root   #model.ckpt=imagenet256_cond
