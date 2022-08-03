
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


## Generates the augmented Lagrangian subproblem
function generate_and_solve_subproblem(instance, scenario, λ, μ, z1, z2, ρ)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    lag_sub = Model(myGurobi)
    set_silent(lag_sub)

    # Decision variables
    @variable(lag_sub, x[I], Bin) # 1 if facility is located at i ∈ I, 0 otherwise.
    @variable(lag_sub, 0 <= y[I] <= bigM) # Capacity decided for facility i ∈ I
    @variable(lag_sub, w[I,J] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(lag_sub, z[J] >= 0) # Shortage in location j ∈ J in scenario s ∈ S

    # Constraints
    # Maximum number of servers
    @constraint(lag_sub, numServers,
        sum(x[i] for i in I) <= N
    )
    
    # Capacity limits: cannot deliver more than capacity decided, 
    #   and only if facility was located
    @constraint(lag_sub, capBal[i in I],
        sum(w[i,j] for j in J) <=  y[i]
    )

    @constraint(lag_sub, capLoc[i in I], 
        sum(w[i,j] for j in J) <= x[i] * bigM
    )
    
    # Demand balance: Demand of active clients must be fulfilled
    @constraint(lag_sub, demBal[j in J],
        sum(w[i,j] for i in I) >= D[j, scenario] - z[j]
    )

    # The two-stage objective function
    FirstStage = @expression(lag_sub, 
        sum(O[i] * x[i] + V[i] * y[i] for i in I) 
    )

    SecondStage = @expression(lag_sub,  
        sum(T[i,j] * w[i,j] for i in I, j in J) +
        sum(U * z[j] for j in J)
    )
    
    @objective(lag_sub, Min, (FirstStage + SecondStage) + 
        sum(λ[i,scenario] * x[i] for i in I) +
        sum(μ[i,scenario] * y[i] for i in I) +
        (ρ/2) * sum((z1[i] - x[i])^2 for i in I) +
        (ρ/2) * sum((z2[i] - y[i])^2 for i in I)
    )
    #%
    optimize!(lag_sub)

    return value.(lag_sub[:x]), value.(lag_sub[:y]), objective_value(lag_sub)
end


function progressive_hedging(ins; max_iter = 200)
    
    I = ins.I 
    S = ins.S
    P = ins.P
    
    k = 1
    ϵ = 0.001
    λ = zeros(length(I), length(S))
    μ = zeros(length(I), length(S))
    x_s = zeros(length(I), length(S))
    y_s = zeros(length(I), length(S))
    z1 = zeros(length(I))
    z2 = zeros(length(I))
    residual = Inf
    residual_s = zeros(length(S))
    LB_aug_s = zeros(length(S))
    LB_aug = -Inf
    ρ = 2.5
    
    start = time()    

    while k <= max_iter && residual > ϵ
        
        for s in ins.S
            x_s[:,s], y_s[:,s], LB_aug_s[s] = generate_and_solve_subproblem(ins, s, λ, μ, z1, z2, ρ)
            residual_s[s] = P[s] * (norm(x_s[:,s] - z1)^2 + norm(y_s[:,s] - z2)^2)
        end 
        
        LB_aug = sum(P[s] * LB_aug_s[s] for s in S)
        residual = sum(residual_s[s] for s in S)

        if residual <= ϵ
            stop = time()
            println("Algorithm converged.")                
            println("\nOptimal found: \n Objective value: $(round(LB_aug, digits=2)) \n Total time: $(round(stop-start, digits=2))s \n Residual: $(round(residual, digits=4))\n")
            return z1, z2
        else    
            
            # z-update
            z1 = sum(P[s] * x_s[:,s] for s in S)
            z2 = sum(P[s] * y_s[:,s] for s in S)

            # dual update    
            for s in S
                λ[:,s] = λ[:,s] + ρ.*(x_s[:,s] - z1)
                μ[:,s] = μ[:,s] + ρ.*(y_s[:,s] - z2)
            end
            
            if k % 5 ==0 
                println("Iter $(k): residual: $(round(residual, digits = 4))") 
            end 

            k = k + 1
        end
    end
    println("Maximum number of iterations exceeded.")
    return z1, z2
end

x_s, y_s = progressive_hedging(instance, max_iter = 500)
