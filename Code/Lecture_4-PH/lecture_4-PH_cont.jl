
include("$(pwd())/instance_generation.jl")

TotalFacilities = 5
TotalClients = 10
TotalScenarios = 50

instance = generate_instance(TotalFacilities, TotalClients, TotalScenarios)

function generate_full_problem_continuous(instance::Instance)
    
    I, J, S, N, P, O, V, U, T, D = unroll_instance(instance)

    # Initialize model
    m = Model(myGurobi)
    
    # Decision variables
    @variable(m, y[I] >= 0)     # Capacity decided for facility i ∈ I
    @variable(m, w[I,J,S] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(m, z[J,S] >= 0)   # Shortage in location j ∈ J in scenario s ∈ S

    # Constraints
    # Capacity limits: cannot deliver more than capacity decided, 
    #   and only if facility was located
    @constraint(m, capBal[i in I, s in S],
        sum(w[i,j,s] for j in J) <=  y[i]
    )

    # Demand balance: Demand of active clients must be fulfilled
    @constraint(m, demBal[j in J, s in S],
        sum(w[i,j,s] for i in I) >= D[j,s] - z[j,s]
    )

    # The two-stage objective function
    FirstStage = @expression(m, 
        sum(V[i] * y[i] for i in I) 
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


fullmodel = generate_full_problem_continuous(instance)
optimize!(fullmodel)

# Examine the solution
@show y_bar = value.(fullmodel[:y])
@show obj = objective_value(fullmodel)


## Generates the augmented Lagrangian subproblem
function generate_and_solve_subproblem(instance, scenario, μ, z2, ρ)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    lag_sub = Model(myGurobi)
    set_silent(lag_sub)

    # Decision variables
    @variable(lag_sub, y[I] >= 0) # Capacity decided for facility i ∈ I
    @variable(lag_sub, w[I,J] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(lag_sub, z[J] >= 0) # Shortage in location j ∈ J in scenario s ∈ S
    
    # Capacity limits: cannot deliver more than capacity decided, 
    #   and only if facility was located
    @constraint(lag_sub, capBal[i in I],
        sum(w[i,j] for j in J) <=  y[i]
    )

    # Demand balance: Demand of active clients must be fulfilled
    @constraint(lag_sub, demBal[j in J],
        sum(w[i,j] for i in I) >= D[j, scenario] - z[j]
    )

    # The two-stage objective function
    FirstStage = @expression(lag_sub, 
        sum(V[i] * y[i] for i in I) 
    )

    SecondStage = @expression(lag_sub,  
        sum(T[i,j] * w[i,j] for i in I, j in J) +
        sum(U * z[j] for j in J)
    )
    
    @objective(lag_sub, Min, 
        (FirstStage + SecondStage) + 
        sum(μ[i,scenario] * y[i] for i in I) +
        (ρ/2) * sum((y[i] - z2[i])^2 for i in I)
    )
    
    optimize!(lag_sub)

    return value.(lag_sub[:y]), objective_value(lag_sub)
end


function progressive_hedging(ins; max_iter = 200)
    
    I = ins.I 
    S = ins.S
    P = ins.P

    k = 1
    ϵ = 0.001
    μ = zeros(length(I), length(S))
    y_s = zeros(length(I), length(S))
    z2 = zeros(length(I))
    residual = Inf
    residual_s = zeros(length(S))
    LB_aug_s = zeros(length(S))
    LB_aug = -Inf
    ρ = 0.25

    start = time()    

    while k <= max_iter && residual > ϵ
        
        for s in S
            y_s[:,s], LB_aug_s[s] = generate_and_solve_subproblem(ins, s, μ, z2, ρ)
            residual_s[s] = P[s] * norm(y_s[:,s] - z2)^2 
        end
        
        LB_aug = sum(P[s] * LB_aug_s[s] for s in S)
        residual = sum(residual_s[s] for s in S)
        
        if residual <= ϵ
            stop = time()
            println("Algorithm converged.")
            println("\nOptimal found: \n Objective value: $(round(LB_aug, digits=2)) \n Total time: $(round(stop-start, digits=2))s \n Residual: $(round(residual, digits=4))\n")
            return z2
        else    

            # z-update
            z2 = sum(P[s] * y_s[:,s] for s in S)

            # dual update
            for s in S
                μ[:,s] = μ[:,s] + ρ.*(y_s[:,s] - z2)
            end
            
            if k % 5 ==0 
                println("Iter $(k): residual: $(round(residual, digits = 4))") 
            end    

            k = k + 1
        end
    end
    println("Maximum number of iterations exceeded.")
    return z2
end

z = progressive_hedging(instance, max_iter = 1000)
