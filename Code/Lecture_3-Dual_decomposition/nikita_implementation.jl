
include("$(pwd())/instance_generation.jl")

TotalFacilities = 5
TotalClients = 10
TotalScenarios = 50

instance = generate_instance(TotalFacilities, TotalClients, TotalScenarios)
fullmodel = generate_full_problem(instance)
optimize!(fullmodel)

# Examine the solutions
@show x_bar = Int.(round.(value.(fullmodel[:x]).data))
@show y_bar = value.(fullmodel[:y])
@show obj = objective_value(fullmodel)


function update_lagrangian_multipliers(λ, μ, gx, gy, UB, LB, k, ϵ)

    α = 2.0

    if k%10 == 0
        α = 0.8α
    end
    
    TotalServers, TotalScenarios = size(gx)
    I = 1:TotalServers
    S = 1:TotalScenarios

    for i in I
        for s in 1:length(S)-1
            sum(gx[i,s]^2 for s in 1:length(S)-1) > ϵ ? λ[i,s] = λ[i,s] + α * (gx[i,s] / sum(gx[i,s1]^2 for s1 in 1:length(S)-1)) : 0.0
            sum(gy[i,s]^2 for s in 1:length(S)-1) > ϵ ? μ[i,s] = μ[i,s] + α * (gy[i,s] / sum(gy[i,s1]^2 for s1 in 1:length(S)-1)) : 0.0
        end
    end
    
    return λ, μ

end   

function update_lagrangian_multipliers_cp(dual_sub)

    optimize!(dual_sub)
    return value.(dual_sub[:λ]), value.(dual_sub[:μ])

end 

function add_cut!(dual_sub, CG, gx, gy)
    
    TotalFacilities, TotalScenarios = size(gx)
    
    I = 1:TotalFacilities
    S = 1:TotalScenarios
    
    λ = dual_sub[:λ]
    μ = dual_sub[:μ]
    θ = dual_sub[:θ]
    
    @constraint(dual_sub, 
        θ <= CG + sum(gx[i,s] * λ[i,s] for i in I, s in S) + sum(gy[i,s] * μ[i,s] for i in I, s in S)
    )

end

function generate_dual_subproblem(ins)
    I = ins.I
    S = ins.S

    dual_sub = Model(()-> Gurobi.Optimizer(GRB_ENV))
    set_silent(dual_sub)
    
    @variable(dual_sub, λ[I,S])
    @variable(dual_sub, μ[I,S])
    @variable(dual_sub, θ)
    @objective(dual_sub, Max, θ)

    return dual_sub
end

function lagrangian_heuristic(ins, x_s, y_s)

    I, J, S, K, P, O, V, U, T, D, bigM = unroll_instance(instance)
        
    y_bar, ind = findmax(y_s.data,dims=2)   
    x_bar = [y_bar[i] > 0 ? 1 : 0 for i in I]
    # y_bar = y_s[:,1]
    # x_bar = x_s[:,1]

    UB = 0.0

    for s in S
        scen_sub = Model(()->Gurobi.Optimizer(GRB_ENV))
        set_silent(scen_sub)
        
        @variable(scen_sub, w[I,J] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
        @variable(scen_sub, z[J] >= 0)   # Shortage in location j ∈ J in scenario s ∈ S

        # Constraints
        # Capacity limits: cannot deliver more than capacity decided, 
        #   and only if facility was located
        @constraint(scen_sub, capBal[i in I],
            sum(w[i,j] for j in J) <=  y_bar[i]
        )

        @constraint(scen_sub, capLoc[i in I], 
            sum(w[i,j] for j in J) <= x_bar[i] * bigM
        )
        
        # Demand balance: Demand of active clients must be fulfilled
        @constraint(scen_sub, demBal[j in J],
            sum(w[i,j] for i in I) >= D[j, s] - z[j]
        )

        # The two-stage objective function
        FirstStage = @expression(scen_sub, 
            sum(O[i] * x_bar[i] + V[i] * y_bar[i] for i in I) 
        )

        SecondStage = @expression(scen_sub,  
        sum(T[i,j] * w[i,j] for i in I, j in J)
        + sum(U * z[j] for j in J)
        )
        
        @objective(scen_sub, Min, P[s] * (FirstStage + SecondStage))
        
        optimize!(scen_sub)
        UB = UB + objective_value(scen_sub)
    end 

    return UB

end

## Benders decomposition

## Generates the main problem
function generate_and_solve_lagrangian_subproblem(instance, λ, μ)
    
    I, J, S, K, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    lag_sub = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(lag_sub)

    # Decision variables
    @variable(lag_sub, x[I,S], Bin) # 1 if facility is located at i ∈ I, 0 otherwise.
    @variable(lag_sub, 0 <= y[I,S] <= bigM) # Capacity decided for facility i ∈ I
    @variable(lag_sub, w[I,J,S] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(lag_sub, z[J,S] >= 0) # Shortage in location j ∈ J in scenario s ∈ S

    # Constraints
    # Maximum number of servers
    @constraint(lag_sub, numServers[s in S],
        sum(x[i,s] for i in I) <= K
    )
    
    # Capacity limits: cannot deliver more than capacity decided, 
    #   and only if facility was located
    @constraint(lag_sub, capBal[i in I, s in S],
        sum(w[i,j,s] for j in J) <=  y[i,s]
    )

    @constraint(lag_sub, capLoc[i in I, s in S], 
        sum(w[i,j,s] for j in J) <= x[i,s] * bigM
    )
    
    # Demand balance: Demand of active clients must be fulfilled
    @constraint(lag_sub, demBal[j in J, s in S],
        sum(w[i,j,s] for i in I) >= D[j,s] - z[j,s]
    )

    # The two-stage objective function
    @objective(lag_sub, Min,
        sum(P[s] * (      
            sum(O[i] * x[i,s] + V[i] * y[i,s] for i in I) +
            sum(T[i,j] * w[i,j,s] for i in I, j in J) + 
            sum(U * z[j,s] for j in J)) 
            for s in S) +
        sum(λ[i,s] * (x[i,s] - x[i,s+1]) for i in I, s in 1:length(S)-1) +     
        sum(μ[i,s] * (y[i,s] - y[i,s+1]) for i in I, s in 1:length(S)-1)
    )

    optimize!(lag_sub)

    CG = sum(P[s] * (      
            sum(O[i] * value(x[i,s]) + V[i] * value(y[i,s]) for i in I, s in S) +
            sum(T[i,j] * value(w[i,j,s]) for i in I, j in J) + 
            sum(U * value(z[j,s]) for j in J)) 
         for s in S) 

    return value.(lag_sub[:x]), value.(lag_sub[:y]), objective_value(lag_sub), CG
end
#%
function lagrangian_decomposition(ins; max_iter = 200)
    k = 1
    ϵ = 0.01
    λ_k = zeros(length(ins.I), length(ins.S))
    μ_k = zeros(length(ins.I), length(ins.S))
    x_s = zeros(length(ins.I), length(ins.S))
    y_s = zeros(length(ins.I), length(ins.S))
    gx = zeros(length(ins.I), length(ins.S))
    gy = zeros(length(ins.I), length(ins.S))
    LB_k = 0.0
    CG_k = 0.0
    LB = -Inf
    UB = Inf
    residual = Inf


    # Bundle method related parameters
 
    CG = Array{Array{Float64}}(undef, max_iter)    # centre of gravity 
    DV_CG = Array{Float64}(undef, max_iter) # dual function value at the centre of gravity 
    cutting_plane_OV = Array{Float64}(undef, max_iter) #cutting plane subproblem objective value

    # Parameters values for the parameters of the Bundle method
    m = 0.6
    d = 100.0
    
    dual_sub = generate_dual_subproblem(ins)


    # generate cutting plane subproblem 
    cutting_plane_subproblem = Model(() -> Gurobi.Optimizer(GRB_ENV)) 
    # set_optimizer_attribute(cutting_plane_subproblem, "print_level", 0)
    set_silent(cutting_plane_subproblem)
    @variables cutting_plane_subproblem begin
        z
        lagrangian_multipliers_representing_variable[ 1:2*length(ins.I), 1:length(ins.S)]
    end

    start = time()    

    while k <= max_iter && residual > ϵ
        
        x_s, y_s, LB_k, CG_k = generate_and_solve_lagrangian_subproblem(ins, λ_k, μ_k)  

        LB = max(LB_k, LB) 

        if k%10 == 0 
            UB_k =  0.0 # lagrangian_heuristic(ins, x_s, y_s)
            UB = min(UB_k, UB)
        end
        
        # Calculate residual
        residual = sqrt(sum((x_s[i,s] - x_s[i,s+1])^2 for i in ins.I, s in 1:length(ins.S)-1) + 
            sum((y_s[i,s] - y_s[i,s+1])^2 for i in ins.I, s in 1:length(ins.S)-1))    
        
        

        if residual <= ϵ
            stop = time()
            println("\nOptimal found. \n Objective value: $(round(LB, digits=2)) 
                                      \n Total time: $(round(stop-start, digits=2))s 
                                      \n Residual: $(round(residual, digits=4))"
            )
            return x_s, y_s, LB
        else    
            
            x_s = x_s.data
            y_s = y_s.data

            for s in 1:length(ins.S)-1
                
                gx[:,s] = x_s[:,s] - x_s[:,s+1]
                gy[:,s] = y_s[:,s] - y_s[:,s+1]
            end       
            
           
            # Making decision regarding NULL step or serious step 
            if k == 1
                CG[k] = [λ_k; μ_k]
                DV_CG[k] = LB
            elseif LB - DV_CG[k-1] >=  m * ((cutting_plane_OV[k-1] + d * sum( norm([λ_k; μ_k][:, s] - CG[k-1][:,s])^2 for s = 1:length(ins.S))) - DV_CG[k-1])
                # serious step
                CG[k] = [λ_k; μ_k]
                DV_CG[k] = LB
                print("Serious Step\n")
            else 
                #null step
                CG[k] = CG[k-1]
                DV_CG[k] = DV_CG[k-1]
            end

            
            #reformulating cutting plabe subproblem
            @objective(cutting_plane_subproblem, Max, cutting_plane_subproblem[:z] - d * sum(  sum((cutting_plane_subproblem[:lagrangian_multipliers_representing_variable][:, s] .- CG[k][:,s]).^2 ) for  s in 1 : length(ins.S) ) )
            
            #adding cut 
            @constraint(cutting_plane_subproblem, cutting_plane_subproblem[:z]  <= DV_CG[k] + sum( sum( [gx[:, s]; gy[:, s]] .* ( cutting_plane_subproblem[:lagrangian_multipliers_representing_variable][:, s] .- [λ_k; μ_k][:, s] ) ) for s = 1 :  length(ins.S) ) )
           
            optimize!(cutting_plane_subproblem)
            #show CG
            #@show cutting_plane_OV[k]

            #@show [λ_k; μ_k]


            for s = 1:length(ins.S)
                λ_k[:,s] = value.(cutting_plane_subproblem[:lagrangian_multipliers_representing_variable])[1:length(ins.I), s]
                μ_k[:,s] = value.(cutting_plane_subproblem[:lagrangian_multipliers_representing_variable])[length(ins.I)+1:2*length(ins.I), s]
            end

            cutting_plane_OV[k] = value.(cutting_plane_subproblem[:z])

            # λ_k, μ_k = update_lagrangian_multipliers(λ_k, μ_k, gx, gy, UB, LB, k, ϵ)
            # add_cut!(dual_sub, CG_k, gx, gy)
            # λ_k, μ_k = update_lagrangian_multipliers_cp(dual_sub)
            
            println("Iter $(k): UB: $(round(UB, digits=2)), LB: $(round(LB, digits=2)), residual: $(round(residual, digits = 4))")

            # tunning the stepsize (d)
            # if (k>0)   d = 10000.0 end
            # if (k>100) d = 1000.0 end
            # if (k>160) d = 100.0 end 
            # if (k>350) d = 10.0 end
            # if (k>400) d = 1.0 end
            # if (k>430) d = 100.0 end
            
            k += 1
        end
    end
    println("Maximum number of iterations exceeded")
    return x_s, y_s, LB
end


x_s, y_s, LB = lagrangian_decomposition(instance, max_iter=500)