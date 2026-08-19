import torch
def repaint_schedule(T, jump_length=10, jump_n_sample=3):
    start_resampling = T-1
    jumps = {
        j: jump_n_sample -1
        for j in range(0, T-jump_length, jump_length)
    } 
    t = T-1
    times = [t]
    while t >=0:
        t-=1
        times.append(t)
        if((jumps.get(t,0) > 0)and t<= start_resampling - jump_length):
            jumps[t] -= 1
            for _ in range(jump_length):
                t +=1
                times.append(t)
    return times

def forward_one_step(x,scheduler, next):
    beta = scheduler.betas[next].to(device=x.device)
    alpha = 1.0 - beta
    z = torch.rand_like(x)
    return(torch.sqrt(alpha)*x + torch.sqrt(beta)*z)

def sample_repaint(model, scheduler, x0, mask, jump_length=10,jump_n_sample=3, steps_to_save = [999,600,300,100,0]):
    with torch.no_grad():
        model.eval()
        B=x0.shape[0]
        T=scheduler.betas.shape[0]
        mask=mask.to(x0.device)

        times = repaint_schedule(T=T, jump_length=jump_length, jump_n_sample=jump_n_sample)
        xt=torch.rand_like(x0)
        snapshots = {}
        save_steps = set(steps_to_save)
        if T-1 in save_steps:
            snapshots[T-1] = xt.detach().clone()

        for current, next in zip(times[:-1], times[1:]):
            if next < current : 

                t=torch.full((B,),current,device=x0.device,dtype=torch.long)
                predicted_noise=model(xt,t)
                xt_unknown,x0_pred=scheduler.reverse_sampling_step(xt,predicted_noise,current)
                if next>=0:
                    gaussian_noise=torch.rand_like(x0)
                    t_prev=torch.full((B,),next,device=x0.device,dtype=torch.long)
                    x_known=scheduler.add_noise(x0,gaussian_noise,t_prev)
                else:
                    x_known=x0
                xt = mask*x_known +(1.0-mask)*xt_unknown

                if next in save_steps:
                    snapshots[next] = xt.detach().clone()
                
                elif next > current:
                    xt = forward_one_step(xt,scheduler, next)
        return xt, snapshots, times