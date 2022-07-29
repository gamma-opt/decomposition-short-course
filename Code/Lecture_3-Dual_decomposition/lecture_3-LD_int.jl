
include("$(pwd())/instance_generation.jl")

TotalFacilities = 10
TotalClients = 50
TotalScenarios = 50

instance = generate_instance(TotalFacilities, TotalClients, TotalScenarios)

function generate_full_problem_binary(instance::Instance)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)

    # Initialize model
    m = Model(myGurobi)
    
    # Decision variables
    @variable(m, x[I], Bin)     # 1 if facility is located at i ∈ I, 0 otherwise.
    @variable(m, w[I,J,S] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(m, z[J,S] >= 0)   # Shortage in location j ∈ J in scenario s ∈ S

    # Constraints
    # Maximum number of servers
    @constraint(m, numServers,
        sum(x[i] for i in I) <= N
    )

    @constraint(m, capLoc[i in I, s in S], 
        sum(w[i,j,s] for j in J) <= x[i] * bigM
    )
    
    # Demand balance: Demand of active clients must be fulfilled
    @constraint(m, demBal[j in J, s in S],
        sum(w[i,j,s] for i in I) >= D[j,s] - z[j,s]
    )

    # The two-stage objective function
    FirstStage = @expression(m, 
        sum(O[i] * x[i] for i in I) 
    )

    SecondStage = @expression(m, 
        sum(P[s] * (
                sum(T[i,j] * w[i,j,s] for i in I, j in J)
                + sum(U * z[j,s] for j in J)) 
        for s in S)
    )
    
    @objective(m, Min, FirstStage + SecondStage)
    
    return m  # Return the generated model
end

fullmodel = generate_full_problem_binary(instance)
optimize!(fullmodel)

# Examine the solutions
@show x_bar = Int.(round.(value.(fullmodel[:x]).data))
@show obj = objective_value(fullmodel)


## Generates the main problem
function generate_and_solve_lagrangian_subproblem(instance, λ)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    lag_sub = Model(myGurobi)
    set_silent(lag_sub)

    # Decision variables
    @variable(lag_sub, x[I,S], Bin) # 1 if facility is located at i ∈ I, 0 otherwise.
    @variable(lag_sub, w[I,J,S] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(lag_sub, z[J,S] >= 0) # Shortage in location j ∈ J in scenario s ∈ S

    # Constraints
    # Maximum number of servers
    @constraint(lag_sub, numServers[s in S],
        sum(x[i,s] for i in I) <= N
    )
    
    # Capacity limits: cannot deliver more than capacity decided, 
    #   and only if facility was located
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
            sum(O[i] * x[i,s] for i in I) +
            sum(T[i,j] * w[i,j,s] for i in I, j in J) + 
            sum(U * z[j,s] for j in J)) 
            for s in S) +
        sum(λ[i,s] * (x[i,s] - x[i,s+1]) for i in I, s in 1:length(S)-1)
    )

    optimize!(lag_sub)

    return value.(lag_sub[:x]), objective_value(lag_sub)
end
#%


function update_lagrangian_multipliers_subgradient(λ, g_x, UB, LB, k, ϵ)

    TotalServers, TotalScenarios = size(g_x)
    I = 1:TotalServers
    S = 1:TotalScenarios
    
    α = 0.1

    if sum(g_x[i,s]^2 for i in I, s in 1:length(S)-1) > ϵ 
        η = α*(UB - LB) / sum(g_x[i,s]^2 for i in I, s in 1:length(S)-1)
    else
        0.0
    end

    λ = λ + η .* g_x 

    return λ

end   


function generate_bundle_subproblem(ins)
    I = ins.I
    S = ins.S

    dual_sub = Model(myGurobi)
    set_silent(dual_sub)
    
    @variable(dual_sub, λ[I,S])
    @variable(dual_sub, θ)

    return dual_sub
end


function update_bundle_subproblem(dual_sub, λ_k, α, g_x, CG_λ, DV_CG)
    
    TotalFacilities, TotalScenarios = size(g_x)
    I = 1:TotalFacilities
    S = 1:TotalScenarios
    
    λ = dual_sub[:λ]
    θ = dual_sub[:θ]
    
    @objective(dual_sub, Max, θ - 
        α * sum((λ[i,s] - CG_λ[i,s])^2 for i in I, s in 1:length(S)-1)
    )
    
    @constraint(dual_sub, θ <= DV_CG + 
        sum(sum(g_x[:,s] .* (λ[:,s] - λ_k[:,s]) for s = 1:length(S)-1))
    )
    
    return dual_sub
end


function update_lagrangian_multipliers_bundle(dual_sub, λ_k, g_x, CG_λ, DV_CG, LB_k)

    m = 0.5
    α = 10.0

    dual_sub = update_bundle_subproblem(dual_sub, λ_k, α, g_x, CG_λ, DV_CG)

    optimize!(dual_sub)

    λ_k = value.(dual_sub[:λ])
    M = objective_value(dual_sub)

    # Serious step test
    if (LB_k - DV_CG) >= m * (M - DV_CG)
        # serious step
        CG_λ = λ_k 
        DV_CG = LB_k
        print("Serious step.\n")
    end

    return dual_sub, λ_k, CG_λ, DV_CG

end 


function lagrangian_heuristic(ins, x_s)

    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(ins)
    
    #Take average among scenarios
    x_bar = sum(P[s] .* x_s[:,s] for s in S) 

    n = 0 # counter to not violate limit N
    for i in shuffle(I)
        if n > N || x_bar[i] < 0.6 # threshold for deciding  
            x_bar[i] = 0
        else  
            x_bar[i] = 1
            n = n + 1
        end    
    end    

    UB = 0.0

    for s in S
        scen_sub = Model(myGurobi)
        set_silent(scen_sub)
        
        @variable(scen_sub, w[I,J] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
        @variable(scen_sub, z[J] >= 0)   # Shortage in location j ∈ J in scenario s ∈ S

        # Constraints
        # Capacity limits: deliver only if facility was located
        @constraint(scen_sub, capLoc[i in I], 
            sum(w[i,j] for j in J) <= x_bar[i] * bigM
        )
        
        # Demand balance: Demand of active clients must be fulfilled
        @constraint(scen_sub, demBal[j in J],
            sum(w[i,j] for i in I) >= D[j, s] - z[j]
        )

        # The two-stage objective function
        FirstStage = @expression(scen_sub, 
            sum(O[i] * x_bar[i] for i in I) 
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



function lagrangian_decomposition(ins; max_iter = 200, method=:bundle, heuristic_frenquency=10)
    k = 1
    ϵ = 0.01
    λ_k = zeros(length(ins.I), length(ins.S))
    x_s = zeros(length(ins.I), length(ins.S))
    g_x = zeros(length(ins.I), length(ins.S))
    UB_k = 0.0 
    LB_k = 0.0
    DV_CG = 0.0 
    LB = -Inf
    UB = Inf
    residual = Inf

    # Bundle method related parameters 
    CG_λ = λ_k   # centre of gravity
   
    # Parameters values for the parameters of the Bundle method

    dual_sub = generate_bundle_subproblem(ins)

    start = time()    

    while k <= max_iter && residual > ϵ  
        x_s, LB_k = generate_and_solve_lagrangian_subproblem(ins, λ_k)  
        x_s = x_s.data
        
        if k == 1
            DV_CG = LB_k # dual function value at the centre of gravity
            UB = lagrangian_heuristic(ins, x_s)
        elseif k % heuristic_frenquency == 0
            UB_k = lagrangian_heuristic(ins, x_s)
            UB = min(UB, UB_k)
        end  

        for s in 1:length(ins.S)-1        
            g_x[:,s] = x_s[:,s] - x_s[:,s+1]
        end  
        
        # Calculate residual
        residual = sqrt(sum(sum(g_x[:,s].^2 for s in 1:length(ins.S)-1)))   
        
        # Check for convergence
        if residual <= ϵ
            stop = time()
            println("\nOptimal found. \n Objective value: $(round(LB, digits=2)) 
                                      \n Total time: $(round(stop-start, digits=2))s 
                                      \n Residual: $(round(residual, digits=4))"
            )
            return x_s, LB_k
        end         
        
        if method == :bundle
            dual_sub, λ_k, CG_λ, DV_CG = 
                update_lagrangian_multipliers_bundle(dual_sub, λ_k, g_x, CG_λ, DV_CG, LB_k)
        elseif method == :subgradient
            λ_k = update_lagrangian_multipliers_subgradient(λ_k, g_x, UB, LB_k, k, ϵ)
        else
            error("Unknown dual upate method")
        end

        println("Iter $(k): UB: $(round(UB, digits=2)), LB: $(round(LB_k, digits=2)), residual: $(round(residual, digits = 4))")        
        k += 1
    end
    println("Maximum number of iterations exceeded")
    return x_s, LB
end


x_s, LB = lagrangian_decomposition(instance, max_iter=50, method = :subgradient)
x_s, LB = lagrangian_decomposition(instance, max_iter=100, method = :bundle)