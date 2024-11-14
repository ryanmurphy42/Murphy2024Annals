# Murphy et al. (2024) Quantifying bead accumulation inside a macrophage population

# This code has two main parts.
# 1) Profile likelihood-based workflow for parameter estimation, identifiability analysis, and prediction analysing total cell number data.
# 2) Propagating forward uncertainity in the profiles to predict the distribution of number of beads per cell.

#######################################################################################
## Initialisation including packages, plot options, filepaths to save outputs

using Plots
using LinearAlgebra
using NLopt
using .Threads
using Interpolations
using Distributions
using Roots
using LaTeXStrings
using CSV
using DataFrames
using DifferentialEquations
using Random
using StatsPlots

# plot options
pyplot() 
fnt = Plots.font("sans-serif", 10) 
global cur_colors = palette(:default) 

mm_to_pts_scaling = 283.46/72;
fig_2_across_size=(75.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_3_across_size=(50.0*mm_to_pts_scaling,33.3*mm_to_pts_scaling);
fig_4_across_size=(37.5*mm_to_pts_scaling,25.0*mm_to_pts_scaling);


# filepaths to save outputs
isdir(pwd() * "\\Figs_SyntheticData\\") || mkdir(pwd() * "\\Figs_SyntheticData") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\Figs_SyntheticData\\"] # location to save figures "\\Fig2\\"] # location to save figures

#######################################################################################
## Mathematical model

a = zeros(3); # beta, initial number of living cells, additive Gaussian standard deviation
α1 = 0.0;  # [hour^-1]
β1 = 1/60; # [hour^-1]
η1 = 10^4; # [hour^-1]
N1 = 8*10^4; # [-]
σ1 = 10^4; # [-]

t_max = 48.0;
t=[0.0;0.0;0.0;24.0;24.0;24.0;48.0;48.0;48.0];# synthetic data measurement times
t_smooth = LinRange(0.0,50.0, 101); # for plotting known values and best-fits

n_max = 10^2; # maximum number of beads per cell

N_living_init = 10^5;

#######################################################################################
## Solving the total cell number

function model_totallivingcells(t,a)
    y=zeros(length(t))
    # a[1] # beta - death rate
    # a[2] # initial cell numbe
    y = a[2]*exp.(-a[1] .* t);
    return y
end

#######################################################################################
## Generate Synthetic data for structured model and total cell number

data0_totallivingcells=model_totallivingcells(t,[β1,N1,σ1]); # generate data using known parameter values to add noise
data0_totallivingcells_smooth = model_totallivingcells(t_smooth,[β1,N1,σ1]); # generate data using known parameter values for plotting

Random.seed!(1234)

data_totallivingcells_dists=[Normal(mi,σ1) for mi in data0_totallivingcells] # define Poisson distribution at each point
data_totallivingcells=[rand(data_dist) for data_dist in data_totallivingcells_dists]; # generate the noisy data

#######################################################################################
### Parameter identifiability analysis - MLE, profiles

    # initial guesses for MLE search
    ββ1 = β1; # death rate
    NN1 = N1; # initial cell number
    σσ1 = σ1; # additive Gaussian standard deviation

    TH=-1.921; #95% confidence interval threshold (for confidence interval for model parameters, and confidence set for model solutions)
    TH_realisations = -2.51 # 97.5 confidence interval threshold (for model realisations)
    THalpha = 0.025; # 97.5% data realisations interval threshold

    function error(data,a)
        y=zeros(length(t))
        y=model_totallivingcells(t,a);
        e=0;
        y[y .< 0] .= zero(eltype(y))
        data_dists=[Normal(mi,a[3]) for mi in y]; 
        e+=sum([loglikelihood(data_dists[i],data[i]) for i in 1:length(data_dists)])
        return e
    end
    
    function fun(a)
        return error(data_totallivingcells,a)
    end

    function optimise(fun,θ₀,lb,ub) 
        tomax = (θ,∂θ) -> fun(θ)
        opt = Opt(:LN_NELDERMEAD,length(θ₀))
        opt.max_objective = tomax
        opt.lower_bounds = lb       # Lower bound
        opt.upper_bounds = ub       # Upper bound
        opt.maxtime = 30.0; # maximum time in seconds
        res = optimize(opt,θ₀)
        return res[[2,1]]
    end


    #######################################################################################
    # MLE
   
    P1_1 = ββ1; # P1 # beta
    P2_1 = NN1; # P2 # initial number of living cells
    P3_1 = σσ1; # P3 # standard deviation of additive Gaussian error model

    θG = [P1_1,P2_1,P3_1] # first guess
    
    # lower and upper bounds for parameter estimation
    P1_lb = 0.0; #1/100;
    P1_ub = 1/20; 
    P2_lb = 10^5*0.6 
    P2_ub = 10^5*1.1 
    P3_lb = 10^2;
    P3_ub = 2*10^4; 

    lb=[P1_lb,P2_lb,P3_lb];
    ub=[P1_ub,P2_ub,P3_ub];

    # MLE optimisation
    (xopt,fopt)  = optimise(fun,θG,lb,ub)

    # storing MLE 
    global fmle=fopt
    global P1mle=xopt[1]
    global P2mle=xopt[2]
    global P3mle=xopt[3]

    #### Plot the total cell number vs the data
   # true model solution
    totallivingcells_true_tsmooth = model_totallivingcells(t_smooth,[β1,N1,σ1]);

    # mle
    totallivingcells_mle = model_totallivingcells(t,[P1mle,P2mle,P3mle]); 
    totallivingcells_mle_tsmooth = model_totallivingcells(t_smooth,[P1mle,P2mle,P3mle]);


    fig_data_totals_mle = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]));
    scatter!(fig_data_totals_mle,t,data_totallivingcells,lw=2,xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}",msw=0,color=:black) # data
    plot!(fig_data_totals_mle,t_smooth,totallivingcells_mle_tsmooth,lw=2,msw=0,color=:magenta) # mle
    plot!(fig_data_totals_mle,t_smooth,totallivingcells_true_tsmooth,lw=2,color=:black,ls=:dash) # true model solution
    display(fig_data_totals_mle)
    savefig(fig_data_totals_mle,filepath_save[1] * "fig_data_totals_mle"   * ".pdf")

    #######################################################################################
    # Profiling (using the MLE as the first guess at each point)

    global fmle_tmp = copy(fmle);
    global P1mle_tmp = copy(P1mle);
    global P2mle_tmp = copy(P2mle);
    global P3mle_tmp = copy(P3mle);

    # PARAMETERS (P1, P2, P3 = (β,N0,σ)
    
    nptss=20;

    #Profile P1

    P1min=lb[1]
    P1max=ub[1]
    P1range_lower=reverse(LinRange(P1min,P1mle,nptss))
    P1range_upper=LinRange(P1mle + (P1max-P1mle)/nptss,P1max,nptss)
    
    P1nrange_lower=zeros(2,nptss)
    llP1_lower=zeros(nptss)
    nllP1_lower=zeros(nptss)
    predict_P1_lower=zeros(1,length(t_smooth),nptss)
    predict_P1_realisations_lower_lq=zeros(1,length(t_smooth),nptss)
    predict_P1_realisations_lower_uq=zeros(1,length(t_smooth),nptss)
    
    P1nrange_upper=zeros(2,nptss)
    llP1_upper=zeros(nptss)
    nllP1_upper=zeros(nptss)
    predict_P1_upper=zeros(1,length(t_smooth),nptss)
    predict_P1_realisations_upper_lq=zeros(1,length(t_smooth),nptss)
    predict_P1_realisations_upper_uq=zeros(1,length(t_smooth),nptss)
    
    # start at mle and increase parameter (upper)
    for i in 1:nptss
        function fun1(aa)
            return error(data_totallivingcells,[P1range_upper[i],aa[1],aa[2]])
        end
        lb1=[lb[2],lb[3]];
        ub1=[ub[2],ub[3]];
        local θG1=[P2mle, P3mle]
        local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
        P1nrange_upper[:,i]=xo[:]
        llP1_upper[i]=fo[1]
        
        modelmean = model_totallivingcells(t_smooth,[P1range_upper[i],P1nrange_upper[1,i],P1nrange_upper[2,i]])
        predict_P1_upper[:,:,i]=modelmean;

        loop_data_dists=[Normal(mi,P1nrange_upper[2,i]) for mi in modelmean];
        predict_P1_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
        predict_P1_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];
        
        if fo > fmle
            println("new MLE P1 upper")
            global fmle_tmp=fo
            global P1mle_tmp=P1range_upper[i]
            global P2mle_tmp=P1nrange_upper[1,i]
            global P3mle_tmp=P1nrange_upper[2,i]
        end
    end
    
    # start at mle and decrease parameter (lower)
    for i in 1:nptss
        function fun1a(aa)
            return error(data_totallivingcells,[P1range_lower[i],aa[1],aa[2]])
        end
        lb1=[lb[2],lb[3]];
        ub1=[ub[2],ub[3]];
        local θG1=[P2mle, P3mle]
        local (xo,fo)=optimise(fun1a,θG1,lb1,ub1)
        P1nrange_lower[:,i]=xo[:]
        llP1_lower[i]=fo[1]

        modelmean = model_totallivingcells(t_smooth,[P1range_lower[i],P1nrange_lower[1,i],P1nrange_lower[2,i]])
        predict_P1_lower[:,:,i]=modelmean;

        loop_data_dists=[Normal(mi,P1nrange_lower[2,i]) for mi in modelmean]; # normal distribution about mean
        predict_P1_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
        predict_P1_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

        if fo > fmle
            println("new MLE P1 lower")
            global fmle_tmp = fo
            global P1mle_tmp=P1range_lower[i]
            global P2mle_tmp=P1nrange_lower[1,i]
            global P3mle_tmp=P1nrange_lower[2,i]
        end
    end
    

    #Profile P2
    P2min=lb[2]
    P2max=ub[2]
    P2range_lower=reverse(LinRange(P2min,P2mle,nptss))
    P2range_upper=LinRange(P2mle + (P2max-P2mle)/nptss,P2max,nptss)
    
    P2nrange_lower=zeros(2,nptss)
    llP2_lower=zeros(nptss)
    nllP2_lower=zeros(nptss)
    predict_P2_lower=zeros(1,length(t_smooth),nptss)
    predict_P2_realisations_lower_lq=zeros(1,length(t_smooth),nptss)
    predict_P2_realisations_lower_uq=zeros(1,length(t_smooth),nptss)

    
    P2nrange_upper=zeros(2,nptss)
    llP2_upper=zeros(nptss)
    nllP2_upper=zeros(nptss)
    predict_P2_upper=zeros(1,length(t_smooth),nptss)
    predict_P2_realisations_upper_lq=zeros(1,length(t_smooth),nptss)
    predict_P2_realisations_upper_uq=zeros(1,length(t_smooth),nptss)

    # start at mle and increase parameter (upper)
    for i in 1:nptss
        function fun2(aa)
            return error(data_totallivingcells,[aa[1],P2range_upper[i],aa[2]])
        end
        lb1=[lb[1],lb[3]];
        ub1=[ub[1],ub[3]];
        local θG1=[P1mle, P3mle]
        local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
        P2nrange_upper[:,i]=xo[:]
        llP2_upper[i]=fo[1]

        modelmean = model_totallivingcells(t_smooth,[P2nrange_upper[1,i],P2range_upper[i],P2nrange_upper[2,i]])
        predict_P2_upper[:,:,i]=modelmean;

        loop_data_dists=[Normal(mi,P2nrange_upper[2,i]) for mi in modelmean]; # normal distribution about mean
        predict_P2_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
        predict_P2_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


        if fo > fmle
            println("new MLE P2 upper")
            global fmle_tmp = fo
            global P1mle_tmp=P2nrange_upper[1,i]
            global P2mle_tmp=P2range_upper[i]
            global P3mle_tmp=P2nrange_upper[2,i]
        end
    end
    
    # start at mle and decrease parameter (lower)
    for i in 1:nptss
        function fun2a(aa)
            return error(data_totallivingcells,[aa[1],P2range_lower[i],aa[2]])
        end
        lb1=[lb[1],lb[3]];
        ub1=[ub[1],ub[3]];
        local θG1=[P1mle, P3mle]
        local (xo,fo)=optimise(fun2a,θG1,lb1,ub1)
        P2nrange_lower[:,i]=xo[:]
        llP2_lower[i]=fo[1]
        
        modelmean = model_totallivingcells(t_smooth,[P2nrange_lower[1,i],P2range_lower[i],P2nrange_lower[2,i]])
        predict_P2_lower[:,:,i]=modelmean;

        loop_data_dists=[Normal(mi,P2nrange_lower[2,i]) for mi in modelmean]; # normal distribution about mean
        predict_P2_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
        predict_P2_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];
        
        if fo > fmle
            println("new MLE S lower")
            global fmle_tmp = fo
            global P1mle_tmp=P2nrange_lower[1,i]
            global P2mle_tmp=P2range_lower[i]
            global P3mle_tmp=P2nrange_lower[2,i]
        end
    end
    


    #Profile P3
    P3min=lb[3]
    P3max=ub[3]
    P3range_lower=reverse(LinRange(P3min,P3mle,nptss))
    P3range_upper=LinRange(P3mle + (P3max-P3mle)/nptss,P3max,nptss)
    
    P3nrange_lower=zeros(2,nptss)
    llP3_lower=zeros(nptss)
    nllP3_lower=zeros(nptss)
    predict_P3_lower=zeros(1,length(t_smooth),nptss)
    predict_P3_realisations_lower_lq=zeros(1,length(t_smooth),nptss)
    predict_P3_realisations_lower_uq=zeros(1,length(t_smooth),nptss)

    P3nrange_upper=zeros(2,nptss)
    llP3_upper=zeros(nptss)
    nllP3_upper=zeros(nptss)
    predict_P3_upper=zeros(1,length(t_smooth),nptss)
    predict_P3_realisations_upper_lq=zeros(1,length(t_smooth),nptss)
    predict_P3_realisations_upper_uq=zeros(1,length(t_smooth),nptss)

    
    # start at mle and increase parameter (upper)
    for i in 1:nptss
        function fun3(aa)
            return error(data_totallivingcells,[aa[1],aa[2],P3range_upper[i]])
        end
        lb1=[lb[1],lb[2]];
        ub1=[ub[1],ub[2]];
        local θG1=[P1mle,P2mle]    
        local (xo,fo)=optimise(fun3,θG1,lb1,ub1)
        P3nrange_upper[:,i]=xo[:]
        llP3_upper[i]=fo[1]

        modelmean = model_totallivingcells(t_smooth,[P3nrange_upper[1,i],P3nrange_upper[2,i],P3range_upper[i]])
        predict_P3_upper[:,:,i]=modelmean;

        loop_data_dists=[Normal(mi,P3range_upper[i]) for mi in modelmean]; # normal distribution about mean
        predict_P3_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
        predict_P3_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

        if fo > fmle
            println("new MLE P3 upper")
            global fmle_tmp = fo
            global P1mle_tmp=P3nrange_upper[1,i]
            global P2mle_tmp=P3nrange_upper[2,i]
            global P3mle_tmp=P3range_upper[i]
        end
    end
    
    # start at mle and decrease parameter (lower)
    for i in 1:nptss
        function fun3a(aa)
            return error(data_totallivingcells,[aa[1],aa[2],P3range_lower[i]])
        end
        lb1=[lb[1],lb[2]];
        ub1=[ub[1],ub[2]];
        local θG1=[P1mle,P2mle]      
        local (xo,fo)=optimise(fun3a,θG1,lb1,ub1)
        P3nrange_lower[:,i]=xo[:]
        llP3_lower[i]=fo[1]
    
        modelmean = model_totallivingcells(t_smooth,[P3nrange_lower[1,i],P3nrange_lower[2,i],P3range_lower[i]])
        predict_P3_lower[:,:,i]=modelmean;

        loop_data_dists=[Normal(mi,P3range_lower[i]) for mi in modelmean]; # normal distribution about mean
        predict_P3_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
        predict_P3_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

        if fo > fmle
            println("new MLE P3 lower")
            global fmle_tmp = fo
            global P1mle_tmp=P3nrange_lower[1,i]
            global P2mle_tmp=P3nrange_lower[2,i]
            global P3mle_tmp=P3range_lower[i]
        end
    end
    

   ## Update MLE guess

   if fmle_tmp > fmle;
    println("Improved estimate of MLE")
    global fmle = copy(fmle_tmp);
    global P1mle = copy(P1mle_tmp);
    global P2mle = copy(P2mle_tmp);
    global P3mle = copy(P3mle_tmp);

   end
    
    ## Predictions

    # Parameter 1 
        # combine the lower and upper
        P1range = [reverse(P1range_lower); P1range_upper]
        P1nrange = [reverse(P1nrange_lower); P1nrange_upper ]
        llP1 = [reverse(llP1_lower); llP1_upper] 
        nllP1=llP1.-maximum(llP1);

        max_P1=zeros(1,length(t_smooth))
        min_P1=100000*ones(1,length(t_smooth))
        for i in 1:(nptss)
            if (llP1_lower[i].-maximum(llP1)) >= TH
                for j in 1:length(t_smooth)
                    max_P1[1,j]=max(predict_P1_lower[1,j,i],max_P1[1,j])
                    min_P1[1,j]=min(predict_P1_lower[1,j,i],min_P1[1,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llP1_upper[i].-maximum(llP1)) >= TH
                for j in 1:length(t_smooth)
                    max_P1[1,j]=max(predict_P1_upper[1,j,i],max_P1[1,j])
                    min_P1[1,j]=min(predict_P1_upper[1,j,i],min_P1[1,j]) 
                end
            end
        end
        # combine the confidence sets for realisations
        max_realisations_P1=zeros(1,length(t_smooth))
        min_realisations_P1=10^10*ones(1,length(t_smooth))
        for i in 1:(nptss)
            if (llP1_lower[i].-maximum(llP1)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_P1[1,j]=max(predict_P1_realisations_lower_uq[1,j,i],max_realisations_P1[1,j])
                    min_realisations_P1[1,j]=min(predict_P1_realisations_lower_lq[1,j,i],min_realisations_P1[1,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llP1_upper[i].-maximum(llP1)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_P1[1,j]=max(predict_P1_realisations_upper_uq[1,j,i],max_realisations_P1[1,j])
                    min_realisations_P1[1,j]=min(predict_P1_realisations_upper_lq[1,j,i],min_realisations_P1[1,j]) 
                end
            end
        end

        # Parameter 2
    # combine the lower and upper
    P2range = [reverse(P2range_lower);P2range_upper]
    P2nrange = [reverse(P2nrange_lower); P2nrange_upper ]
    llP2 = [reverse(llP2_lower); llP2_upper]     
    nllP2=llP2.-maximum(llP2);
    
    max_P2=zeros(1,length(t_smooth))
    min_P2=10^10*ones(1,length(t_smooth))
    for i in 1:(nptss)
        if (llP2_lower[i].-maximum(llP2)) >= TH
            for j in 1:length(t_smooth)
                max_P2[1,j]=max(predict_P2_lower[1,j,i],max_P2[1,j])
                min_P2[1,j]=min(predict_P2_lower[1,j,i],min_P2[1,j])
            end
        end
    end
    for i in 1:(nptss)
        if (llP2_upper[i].-maximum(llP2)) >= TH
            for j in 1:length(t_smooth)
                max_P2[1,j]=max(predict_P2_upper[1,j,i],max_P2[1,j])
                min_P2[1,j]=min(predict_P2_upper[1,j,i],min_P2[1,j]) 
            end
        end
    end
    # combine the confidence sets for realisations
    max_realisations_P2=zeros(1,length(t_smooth))
    min_realisations_P2=10^10*ones(1,length(t_smooth))
    for i in 1:(nptss)
        if (llP2_lower[i].-maximum(llP2)) >= TH_realisations
            for j in 1:length(t_smooth)
                max_realisations_P2[1,j]=max(predict_P2_realisations_lower_uq[1,j,i],max_realisations_P2[1,j])
                min_realisations_P2[1,j]=min(predict_P2_realisations_lower_lq[1,j,i],min_realisations_P2[1,j])
            end
        end
    end
    for i in 1:(nptss)
        if (llP2_upper[i].-maximum(llP2)) >= TH_realisations
            for j in 1:length(t_smooth)
                max_realisations_P2[1,j]=max(predict_P2_realisations_upper_uq[1,j,i],max_realisations_P2[1,j])
                min_realisations_P2[1,j]=min(predict_P2_realisations_upper_lq[1,j,i],min_realisations_P2[1,j]) 
             end
        end
    end
    
        # Parameter 3 
    # combine the lower and upper
    P3range = [reverse(P3range_lower);P3range_upper]
    P3nrange = [reverse(P3nrange_lower); P3nrange_upper ]
    llP3 = [reverse(llP3_lower); llP3_upper] 
    
    nllP3=llP3.-maximum(llP3)

    max_P3=zeros(1,length(t_smooth))
    min_P3=10^10*ones(1,length(t_smooth))
    for i in 1:(nptss)
        if (llP3_lower[i].-maximum(llP3)) >= TH_realisations
            for j in 1:length(t_smooth)
                max_P3[1,j]=max(predict_P3_lower[1,j,i],max_P3[1,j])
                min_P3[1,j]=min(predict_P3_lower[1,j,i],min_P3[1,j])
              end
        end
    end
    for i in 1:(nptss)
        if (llP3_upper[i].-maximum(llP3)) >= TH_realisations
            for j in 1:length(t_smooth)
                max_P3[1,j]=max(predict_P3_upper[1,j,i],max_P3[1,j])
                min_P3[1,j]=min(predict_P3_upper[1,j,i],min_P3[1,j]) 
           end
        end
    end
        # combine the confidence sets for realisations
        max_realisations_P3=zeros(1,length(t_smooth))
        min_realisations_P3=10^10*ones(1,length(t_smooth))
        for i in 1:(nptss)
            if (llP3_lower[i].-maximum(llP3)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_P3[1,j]=max(predict_P3_realisations_lower_uq[1,j,i],max_realisations_P3[1,j])
                    min_realisations_P3[1,j]=min(predict_P3_realisations_lower_lq[1,j,i],min_realisations_P3[1,j])
                 end
            end
        end
        for i in 1:(nptss)
            if (llP3_upper[i].-maximum(llP3)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_P3[1,j]=max(predict_P3_realisations_upper_uq[1,j,i],max_realisations_P3[1,j])
                    min_realisations_P3[1,j]=min(predict_P3_realisations_upper_lq[1,j,i],min_realisations_P3[1,j]) 
                end
            end
        end


    
        ##############################################################################################################
        # Updated mle plot
        totallivingcells_mle_update = model_totallivingcells(t,[P1mle,P2mle,P3mle]); 
        totallivingcells_mle_tsmooth_update = model_totallivingcells(t_smooth,[P1mle,P2mle,P3mle]);

        fig_data_totals_mle_update = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]));
        scatter!(fig_data_totals_mle_update,t,data_totallivingcells,lw=2,xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}",msw=0,color=:black) # data
        plot!(fig_data_totals_mle_update,t_smooth,totallivingcells_mle_tsmooth_update,lw=2,msw=0,color=:magenta) # mle
        plot!(fig_data_totals_mle_update,t_smooth,totallivingcells_true_tsmooth,lw=2,color=:black,ls=:dash) # true model solution
        display(fig_data_totals_mle_update)
        savefig(fig_data_totals_mle_update,filepath_save[1] * "fig_data_totals_mle_update"   * ".pdf")

    
        ##############################################################################################################
        # Plot Profile likelihoods
        
        combined_plot_linewidth = 2;

        # interpolate for smoother profile likelihoods
        interp_nptss= 1001;
        
        # β1
        interp_points_P1range =  LinRange(P1min,P1max,interp_nptss)
        interp_P1 = LinearInterpolation(P1range,nllP1)
        interp_nllP1 = interp_P1(interp_points_P1range)
        
        profile1=plot(interp_points_P1range,interp_nllP1,xlim=(P1min,P1max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"\beta \ [\mathrm{hr}^{-1}]",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3,size=fig_2_across_size)
        profile1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile1=vline!([P1mle],lw=3,linecolor=:red)
        profile1=vline!([ β1 ],lw=3,linecolor=:rosybrown,linestyle=:dash)


        # N1(0)
        interp_points_P2range =  LinRange(P2min,P2max,interp_nptss)
        interp_P2 = LinearInterpolation(P2range,nllP2)
        interp_nllP2 = interp_P2(interp_points_P2range)

        profile2=plot(interp_points_P2range,interp_nllP2,xlim=(P2min,P2max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xticks=([6*10^4,8*10^4,10*10^4],[6,8,10]),xlab=L"\quad \quad \quad  \quad \quad \quad \quad N(0) \ [-] \quad \quad \quad \quad \times 10^{4}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3,size=fig_2_across_size)
        profile2=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile2=vline!([P2mle],lw=3,linecolor=:red)
        profile2=vline!([NN1],lw=3,linecolor=:rosybrown,linestyle=:dash)


        # σ
        interp_points_P3range =  LinRange(P3min,P3max,interp_nptss)
        interp_P3 = LinearInterpolation(P3range,nllP3)
        interp_nllP3 = interp_P3(interp_points_P3range)
        
        profile3=plot(interp_points_P3range,interp_nllP3,xlim=(P3min,P3max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xticks=[5*10^3,10*10^3,15*10^3],xlab=L"\sigma \ [-]",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3,size=fig_2_across_size)
        profile3=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile3=vline!([P3mle],lw=3,linecolor=:red)
        profile3=vline!([σ1],lw=3,linecolor=:rosybrown,linestyle=:dash)


            
        display(profile1)
        savefig(profile1,filepath_save[1] * "Fig_profile1"    * ".pdf")
        
        display(profile2)
        savefig(profile2,filepath_save[1] * "Fig_profile2"    * ".pdf")
        
        display(profile3)
        savefig(profile3,filepath_save[1] * "Fig_profile3"  * ".pdf")
        

        #######################################################################################
        # compute the bounds of confidence interval
        
        function fun_interpCI(mle,interp_points_range,interp_nll,TH)
            # find bounds of CI
            range_minus_mle = interp_points_range - mle*ones(length(interp_points_range),1)
            abs_range_minus_mle = broadcast(abs, range_minus_mle)
            findmin_mle = findmin(abs_range_minus_mle)
        
            # find closest value to CI threshold intercept
            value_minus_threshold = interp_nll - TH*ones(length(interp_nll),1)
            abs_value_minus_threshold = broadcast(abs, value_minus_threshold)
            lb_CI_tmp = findmin(abs_value_minus_threshold[1:findmin_mle[2][1]])
            ub_CI_tmp = findmin(abs_value_minus_threshold[findmin_mle[2][1]:length(abs_value_minus_threshold)])
            lb_CI = interp_points_range[lb_CI_tmp[2][1]]
            ub_CI = interp_points_range[findmin_mle[2][1]-1 + ub_CI_tmp[2][1]]
        
            return lb_CI,ub_CI
        end
        
        # β 
        (lb_CI_P1,ub_CI_P1) = fun_interpCI(P1mle,interp_points_P1range,interp_nllP1,TH)
        println(round(lb_CI_P1; digits = 4))
        println(round(ub_CI_P1; digits = 4))
        
        # N(0)
        (lb_CI_P2,ub_CI_P2) = fun_interpCI(P2mle,interp_points_P2range,interp_nllP2,TH)
        println(round(lb_CI_P2; digits = 3))
        println(round(ub_CI_P2; digits = 3))
        
        # σ
        (lb_CI_P3,ub_CI_P3) = fun_interpCI(P3mle,interp_points_P3range,interp_nllP3,TH)
        println(round(lb_CI_P3; digits = 3))
        println(round(ub_CI_P3; digits = 3))


        # Export MLE and bounds to csv (one file for all data) -- -MLE ONLY 
        if @isdefined(df_MLEBoundsAll) == 0
            println("not defined")
            global df_MLEBoundsAll = DataFrame(beta1mle=P1mle, lb_CI_beta1=lb_CI_P1, ub_CI_beta1=ub_CI_P1, N0mle0=P2mle, lb_CI_N0=lb_CI_P2, ub_CI_N0=ub_CI_P2, sigmamle=P3mle, lb_CI_sigma=lb_CI_P3, ub_CI_sigma=ub_CI_P3 )
        else 
            println("DEFINED")
            global df_MLEBoundsAll_thisrow = DataFrame(beta1mle=P1mle, lb_CI_beta1=lb_CI_P1, ub_CI_beta1=ub_CI_P1, N0mle0=P2mle, lb_CI_N0=lb_CI_P2, ub_CI_N0=ub_CI_P2, sigmamle=P3mle, lb_CI_sigma=lb_CI_P3, ub_CI_sigma=ub_CI_P3)
            append!(df_MLEBoundsAll,df_MLEBoundsAll_thisrow)
        end
        
        CSV.write(filepath_save[1] * "MLEBoundsALL.csv", df_MLEBoundsAll)




        #########################################################################################
        # Confidence sets for model solutions
        colour1 = :magenta;
        ymax=1.2*10^5;

        ymle_smooth = model_totallivingcells(t_smooth,[P1mle,P2mle,P3mle]);

           # P1
           confmodel1 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
           plot!(confmodel1,t_smooth,ymle_smooth,lw=3,linecolor=colour1) # solid - c1
           plot!(confmodel1,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_P1[1,:],max_P1[1,:].-ymle_smooth),fillalpha=.2)
           display(confmodel1)
           savefig(confmodel1,filepath_save[1] * "Fig_confmodel1"   * ".pdf")
   
           # P2
           confmodel2 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
           plot!(confmodel2,t_smooth,ymle_smooth,lw=3,linecolor=colour1) # solid - c1
           plot!(confmodel2,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_P2[1,:],max_P2[1,:].-ymle_smooth),fillalpha=.2)
           display(confmodel2)
           savefig(confmodel2,filepath_save[1] * "Fig_confmodel2"   * ".pdf")

           # P3
           confmodel3 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
           plot!(confmodel3,t_smooth,ymle_smooth,lw=3,linecolor=colour1) # solid - c1
           plot!(confmodel3,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_P3[1,:],max_P3[1,:].-ymle_smooth),fillalpha=.2)
           display(confmodel3)
           savefig(confmodel3,filepath_save[1] * "Fig_confmodel3"   * ".pdf")
   
   
           # union
           max_overall=zeros(1,length(t_smooth))
           min_overall=1000*ones(1,length(t_smooth))
           for j in 1:length(t_smooth)
               max_overall[1,j]=max(max_P1[1,j],max_P2[1,j],max_P3[1,j])
                min_overall[1,j]=min(min_P1[1,j],min_P2[1,j],min_P3[1,j])
            end
            confmodelu = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
           plot!(confmodelu,t_smooth,ymle_smooth,lw=2,linecolor=colour1) # solid - c1
                plot!(confmodelu,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_overall[1,:],max_overall[1,:].-ymle_smooth),fillalpha=.2)

            display(confmodelu)
            savefig(confmodelu,filepath_save[1] * "Fig_confmodelu"   * ".pdf")

            
#########################################################################################
        # Confidence sets for model realisations
        


   # P1
   confreal1 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
    plot!(confreal1,t_smooth,ymle_smooth,lw=3,linecolor=colour1) # solid - c1
   scatter!(confreal1,t,data_totallivingcells,legend=false,markercolor=:black,msw=0) # scatter c1
   plot!(confreal1,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_realisations_P1[1,:],max_realisations_P1[1,:].-ymle_smooth),fillalpha=.2)
   display(confreal1)
   savefig(confreal1,filepath_save[1] * "Fig_confreal1"   * ".pdf")

   # P2
   confreal2 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
   plot!(confreal2,t_smooth,ymle_smooth,lw=3,linecolor=colour1) # solid - c1
    scatter!(confreal2,t,data_totallivingcells,legend=false,markercolor=:black,msw=0) # scatter c1
    plot!(confreal2,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_realisations_P2[1,:],max_realisations_P2[1,:].-ymle_smooth),fillalpha=.2)
   display(confreal2)
   savefig(confreal2,filepath_save[1] * "Fig_confreal2"   * ".pdf")

   # P3
   confreal3 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
   plot!(confreal3,t_smooth,ymle_smooth,lw=3,linecolor=colour1) # solid - c1
   scatter!(confreal3,t,data_totallivingcells,legend=false,markercolor=:black,msw=0) # scatter c1
   plot!(confreal3,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_realisations_P3[1,:],max_realisations_P3[1,:].-ymle_smooth),fillalpha=.2)
   display(confreal3)
   savefig(confreal3,filepath_save[1] * "Fig_Pconfreal3"   * ".pdf")



     # union 
   max_realisations_overall=zeros(2,length(t_smooth))
   min_realisations_overall=1000*ones(2,length(t_smooth))
   for j in 1:length(t_smooth)
       max_realisations_overall[1,j]=max(max_realisations_P1[1,j],max_realisations_P2[1,j],max_realisations_P3[1,j])
        min_realisations_overall[1,j]=min(min_realisations_P1[1,j],min_realisations_P2[1,j],min_realisations_P3[1,j])
     end

     confrealu = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,1.2*10^5],legend=false,size=fig_2_across_size,xlim=[-1,50],xticks=[0,24,48],yticks=([0,5*10^4,1.0*10^5],[0,5,10]),xlab=L"t \ [\mathrm{hr}]",ylab=L"\quad \quad \quad N(t) \quad \quad \times 10^{4}");
    plot!(confrealu,t_smooth,ymle_smooth,lw=3,linecolor=colour1) # solid - c1
   scatter!(confrealu,t,data_totallivingcells,legend=false,markercolor=:black,msw=0) # scatter c1
    plot!(confrealu,t_smooth,ymle_smooth,w=0,c=colour1,ribbon=(ymle_smooth.-min_realisations_overall[1,:],max_realisations_overall[1,:].-ymle_smooth),fillalpha=.2)
  
   display(confrealu)
   savefig(confrealu,filepath_save[1] * "Fig_confrealu"   * ".pdf")


#########################################################################################
#########################################################################################
#########################################################################################
        # Prediction for distribuition of beads per cell
   
        
function DE_structured!(du,u,p,t)
    α, β, η = p
    # number density of live macrophages that contain n >= 0 beads at time t >= 0
    if α == 0.0
        for j=0:n_max
            du[j+1] = η*( sum([ u[n_max+2+k]*u[j+1-k] for k=0:j])  ) - η*u[j+1]*(sum([u[k] for k=(n_max+2):(2*n_max+2)])) - β*u[j+1];
        end
    else
        for j=0:n_max
            du[j+1] = η*( sum([ u[n_max+2+k]*u[j+1-k] for k=0:j])  ) - η*u[j+1]*(sum([u[k] for k=(n_max+2):(2*n_max+2)])) - β*u[j+1]  + 2*α*( sum([ (binomial(k,j))*u[k+1]/2^k for k=j:n_max])  ) - α*u[j+1];
        end
    end
    # number density of apoptotic macrophages that contain n >= 0 beads at time t >= 0
    for j=0:n_max
        du[n_max+2+j] = -η*u[n_max+j+2]*(sum([u[k] for k=1:(n_max+1)])) + β*u[j+1]; 
    end
end

#######################################################################################
# ## Solving the structured mathematical model

function odesolver_structured(t,α,β,η,C0)
    p=[α,β,η]
    tspan=(0.0,maximum(t));
    prob=ODEProblem(DE_structured!,C0,tspan,p);
    sol=solve(prob,saveat=t);
    return sol[1:(2*n_max+2),:];
end

function model_structured(t,a)
    y=zeros(length(t))
    y=N_living_init.*odesolver_structured(t,a[1],a[2],a[3],C0)
    return y
end

function model_structured_total(t,a)
    y_tmp=zeros(length(t))
    y_tmp=N_living_init.*odesolver(t,a[1],a[2],a[3],C0)
    y = [sum(y_tmp[:,1]) for i=1:size(y_tmp,2)];
    return y
end


### Initial condition
filepathseparator = "/"
filepath_load = pwd()* filepathseparator ;
datafiles = ["HughReDigitisedData_t0_V3.csv","HughReDigitisedData_t24_V3.csv","HughReDigitisedData_t48_V3.csv"];
lendatafiles = length(datafiles);
exp_data = [CSV.read([filepath_load * datafiles[i]], DataFrame) for i=1:lendatafiles]

C0 = [exp_data[1][:,2]./sum(exp_data[1][:,2]);zeros(n_max+1-length(exp_data[1][:,1]));zeros(n_max+1)]; # all cells 
# C24 = [exp_data[2][1:94,2]./sum(exp_data[2][1:94,2]);zeros(n_max+1-length(exp_data[2][1:94,1]));zeros(n_max+1)]; # all cells 
# C48 = [exp_data[3][1:97,2]./sum(exp_data[3][1:97,2]);zeros(n_max+1-length(exp_data[3][1:97,1]));zeros(n_max+1)]; # all cells 


t_structured = [0.0,24.0,48.0];

data0_structured=model_structured(t_structured,[α1, β1, η1 ]); # generate data using known parameter values to add noise
data0_structured_smooth = model_structured(t_smooth,[α1, β1, η1 ]); # generate data using known parameter values for plotting


#######################################################################################
# ## Plot the synthetic data, with β=1/60, and MLE and intervals

# compute approximate confidence intervals based on the profile likelihood confidence intervals

    # Parameter 1
    # combine the lower and upper  
    max_P1_structured=zeros((n_max+1),length(t_structured)); # number of compartments x number of timepoints
    min_P1_structured=10^10*ones((n_max+1),length(t_structured)); # number of compartments x number of timepoints
    for i in 1:(nptss)
        if (llP1_lower[i].-maximum(llP1)) >= TH
            sol_structured_tmp = model_structured(t_structured,[α1, P1range_lower[i], η1 ]); 
            for j in 1:length(t_structured)
                for r=1:(n_max+1)
                    max_P1_structured[r,j]=max(sol_structured_tmp[r,j]./sum(sol_structured_tmp[:,j]),max_P1_structured[r,j]);
                    min_P1_structured[r,j]=min(sol_structured_tmp[r,j]./sum(sol_structured_tmp[:,j]),min_P1_structured[r,j]);
                end
            end
        end
    end

    for i in 1:(nptss)
        if (llP1_upper[i].-maximum(llP1)) >= TH
            sol_structured_tmp=model_structured(t_structured,[α1, P1range_upper[i], η1 ]); 
            for j in 1:length(t_structured)
                for r=1:(n_max+1)
                    max_P1_structured[r,j]=max(sol_structured_tmp[r,j]./sum(sol_structured_tmp[:,j]),max_P1_structured[r,j])
                    min_P1_structured[r,j]=min(sol_structured_tmp[r,j]./sum(sol_structured_tmp[:,j]),min_P1_structured[r,j]) 
                end
            end
        end
    end


mle_structured=model_structured(t_structured,[α1, P1mle, η1 ]); 



for k=1:length(t_structured)
    fig_logproportion = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlabel="# beads per cell",ylabel=L"\log_{10}(\text{Proportion})",legend=false,size=fig_2_across_size);
    
     # Plot MLE
     data_mle= copy(mle_structured);
     data_mle_tmp = copy(data_mle[1:(n_max+1),k]./sum(data_mle[1:(n_max+1),k]));
     plot!(fig_logproportion,([0:(n_max);]),log10.(data_mle_tmp),msw=0,ylim=(-5,0),c=colour1,lw=2);
   

    # Compute intervals
    proportion_mle = log10.(data_mle_tmp);
    proportion_mle[ proportion_mle .< -100] .= -200;
    proportion_min = log10.(min_P1_structured[:,k]);
    proportion_min[ proportion_min .< -100] .= -200;
    proportion_max = log10.(max_P1_structured[:,k]);
    proportion_max[ proportion_max .< -100] .= -200;
    proportion_l = proportion_mle - proportion_min;
    proportion_u = proportion_max - proportion_mle;
    plot!(fig_logproportion,([0:(n_max);]),log10.(data_mle_tmp),w=0.1,c=colour1,ribbon=(proportion_l,proportion_u),fillalpha=.2)

   

    # Plot data
    data= copy(data0_structured);
    data_tmp = copy(data[1:(n_max+1),k]./sum(data[1:(n_max+1),k]));
    plot!(fig_logproportion,([0:(n_max);]),log10.(data_tmp),msw=0,ylim=(-5,0),color=:black,ls=:dot,lw=2);

    display(fig_logproportion)
    savefig(fig_logproportion,filepath_save[1] * "fig_logproportion_t_" * string(k)  * ".pdf")

end


