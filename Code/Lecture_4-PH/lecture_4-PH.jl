
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
    k = 1
    ϵ = 0.01
    λ = zeros(length(ins.I), length(ins.S))
    μ = zeros(length(ins.I), length(ins.S))
    x_s = zeros(length(ins.I), length(ins.S))
    y_s = zeros(length(ins.I), length(ins.S))
    z1 = zeros(length(ins.I))
    z2 = zeros(length(ins.I))
    LB = 0.0
    LB_s = 0.0 
    residual = Inf
    ρ = 0.5
    
    start = time()    

    while k <= max_iter && residual > ϵ
        
        for s in ins.S
            x_s[:,s], y_s[:,s] = generate_and_solve_subproblem(ins, s, λ, μ, z1, z2, ρ)
        end
    
        # Calculate residual
        residual = sqrt(sum(ins.P[s]*(x_s[i,s] - z1[i])^2 for i in ins.I, s in ins.S) + 
            sum(ins.P[s]*(y_s[i,s] - z2[i])^2 for i in ins.I, s in ins.S))  
        
        if residual <= ϵ
            stop = time()
            println("Algorithm converged. Calculating bound...\n")
            #Calculate bound 
            for s in ins.S
                x_s[:,s], y_s[:,s], LB_s = generate_and_solve_subproblem(ins, s, λ, μ, z1, z2, 0.0)
                LB = LB + ins.P[s] * LB_s
            end
                
            println("\nOptimal found: \n Objective value: $(round(LB, digits=2)) 
                                      \n Total time: $(round(stop-start, digits=2))s 
                                      \n Residual: $(round(residual, digits=4))"
            )
            return x_s, y_s
        else    

            z1 = sum(ins.P[s] * x_s[:,s] for s in ins.S)
            z2 = sum(ins.P[s] * y_s[:,s] for s in ins.S)

            for s in ins.S
                λ[:,s] = λ[:,s] + ρ.*(x_s[:,s] - z1)
                μ[:,s] = μ[:,s] + ρ.*(y_s[:,s] - z2)
            end
            
            println("Iter $(k): residual: $(round(residual, digits = 4))")

            k += 1
        end
    end
    println("Maximum number of iterations exceeded.")
    return x_s, y_s
end

x_s, y_s = progressive_hedging(instance, max_iter = 300)

sum(instance.P[s].* x_s[:,s] for s in instance.S)
sum(instance.P[s].* y_s[:,s] for s in instance.S)
