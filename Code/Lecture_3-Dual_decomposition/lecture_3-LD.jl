
include("$(pwd())/instance_generation.jl")

TotalFacilities = 5
TotalClients = 10
TotalScenarios = 30

instance = generate_instance(TotalFacilities, TotalClients, TotalScenarios)
fullmodel = generate_full_problem(instance)
optimize!(fullmodel)

# Examine the solutions
@show x_bar = Int.(round.(value.(fullmodel[:x]).data))
@show y_bar = value.(fullmodel[:y])
@show obj = objective_value(fullmodel)


function update_lagrangian_multipliers_subgradient(λ, μ, gx, gy, UB, LB, k, ϵ)

    α_x = 2.0
    α_y = 2.0

    if k%10 == 0
        α_y = 0.8α_y
    end
    
    TotalServers, TotalScenarios = size(gx)
    I = 1:TotalServers
    S = 1:TotalScenarios

    for i in I
        for s in 1:length(S)-1
            sum(gx[i,s]^2 for s in 1:length(S)-1) > ϵ ? λ[i,s] = λ[i,s] + α_x * (gx[i,s] / sum(gx[i,s1]^2 for s1 in 1:length(S)-1)) : 0.0
            sum(gy[i,s]^2 for s in 1:length(S)-1) > ϵ ? μ[i,s] = μ[i,s] + α_y * (gy[i,s] / sum(gy[i,s1]^2 for s1 in 1:length(S)-1)) : 0.0
        end
    end
    
    return λ, μ

end   

function update_lagrangian_multipliers_cp(dual_sub)

    optimize!(dual_sub)
    return value.(dual_sub[:λ]), value.(dual_sub[:μ])

end 

function update_dual_subproblem(dual_sub, λ_k, μ_k, d_x, d_y, gx, gy, CG_λ, CG_μ, DV_CG)
    
    TotalFacilities, TotalScenarios = size(gx)
    
    I = 1:TotalFacilities
    S = 1:TotalScenarios
    
    λ = dual_sub[:λ]
    μ = dual_sub[:μ]
    θ = dual_sub[:θ]
    
    @objective(dual_sub, Max, θ - (
        d_x * (sum(sum((λ[:,s] .- CG_λ[:,s]).^2) for s in 1:length(S)-1)) +
        d_y * (sum(sum((μ[:,s] .- CG_μ[:,s]).^2) for s in 1:length(S)-1)))
    )
    
    @constraint(dual_sub, θ <= DV_CG + 
        sum(sum(gx[:,s] .* (λ[:,s] .- λ_k[:,s]) for s = 1:length(S)-1)) +
        sum(sum(gy[:,s] .* (μ[:,s] .- μ_k[:,s]) for s = 1:length(S)-1))
    )
    
    return dual_sub
end

function generate_dual_subproblem(ins)
    I = ins.I
    S = ins.S

    dual_sub = Model(myGurobi)
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
    UB = 0.0

    for s in S
        scen_sub = Model(myGurobi)
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
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    lag_sub = Model(myGurobi)
    set_silent(lag_sub)

    # Decision variables
    @variable(lag_sub, x[I,S], Bin) # 1 if facility is located at i ∈ I, 0 otherwise.
    @variable(lag_sub, 0 <= y[I,S] <= bigM) # Capacity decided for facility i ∈ I
    @variable(lag_sub, w[I,J,S] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(lag_sub, z[J,S] >= 0) # Shortage in location j ∈ J in scenario s ∈ S

    # Constraints
    # Maximum number of servers
    @constraint(lag_sub, numServers[s in S],
        sum(x[i,s] for i in I) <= N
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

    return value.(lag_sub[:x]), value.(lag_sub[:y]), objective_value(lag_sub)
end
#%

function lagrangian_heuristic(ins, x_s, y_s)

    I, J, S, K, P, O, V, U, T, D, bigM = unroll_instance(instance)
        
    y_bar, ind = findmax(y_s.data,dims=2)   
    x_bar = [y_bar[i] > 0 ? 1 : 0 for i in I]
    UB = 0.0

    for s in S
        scen_sub = Model(myGurobi)
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


function lagrangian_decomposition(ins; max_iter = 200)
    k = 1
    ϵ = 0.01
    λ_k = zeros(length(ins.I), length(ins.S))
    μ_k = zeros(length(ins.I), length(ins.S))
    x_s = zeros(length(ins.I), length(ins.S))
    y_s = zeros(length(ins.I), length(ins.S))
    g_x = zeros(length(ins.I), length(ins.S))
    g_y = zeros(length(ins.I), length(ins.S))
    LB_k = 0.0
    LB = -Inf
    UB = Inf
    residual = Inf

    # Bundle method related parameters 
    CG_λ = λ_k 
    CG_μ = μ_k
    DV_CG = 0.0  # dual function value at the centre of gravity 
    
    # Parameters values for the parameters of the Bundle method
    m = 0.9999
    d_x = 500.0
    d_y = 10000.0
    dual_sub = generate_dual_subproblem(ins)

    start = time()    

    while k <= max_iter && residual > ϵ  
        x_s, y_s, LB_k = generate_and_solve_lagrangian_subproblem(ins, λ_k, μ_k)  

        if k == 1
            DV_CG = LB_k
        end  
                
        # Calculate residual
        residual = sqrt(sum((x_s[i,s] - x_s[i,s+1])^2 for i in ins.I, s in 1:length(ins.S)-1) + 
            sum((y_s[i,s] - y_s[i,s+1])^2 for i in ins.I, s in 1:length(ins.S)-1))   
        
        # Check for convergence
        if residual <= ϵ
            stop = time()
            println("\nOptimal found. \n Objective value: $(round(LB, digits=2)) 
                                      \n Total time: $(round(stop-start, digits=2))s 
                                      \n Residual: $(round(residual, digits=4))"
            )
            return x_s, y_s, LB
        end   
        
        # Update multipliers
        x_s = x_s.data
        y_s = y_s.data

        for s in 1:length(ins.S)-1        
            g_x[:,s] = x_s[:,s] - x_s[:,s+1]
            g_y[:,s] = y_s[:,s] - y_s[:,s+1]
        end       
        
        dual_sub = update_dual_subproblem(dual_sub, λ_k, μ_k, d_x, d_y, g_x, g_y, CG_λ, CG_μ, DV_CG)
        
        optimize!(dual_sub)
        
        λ_k = value.(dual_sub[:λ])
        μ_k = value.(dual_sub[:μ])
        θ = value.(dual_sub[:θ])

        # Making decision regarding NULL step or serious step 
       if LB_k - DV_CG >=  m * (θ - DV_CG)
            # serious step
            CG_λ = λ_k 
            CG_μ = μ_k
            DV_CG = LB_k
            print("Serious step.\n")
        end
        
        println("Iter $(k): UB: $(round(UB, digits=2)), LB: $(round(LB_k, digits=2)), residual: $(round(residual, digits = 4))")        
        k += 1
    end
    println("Maximum number of iterations exceeded")
    return x_s, y_s, LB
end


x_s, y_s, LB = lagrangian_decomposition(instance, max_iter=500)