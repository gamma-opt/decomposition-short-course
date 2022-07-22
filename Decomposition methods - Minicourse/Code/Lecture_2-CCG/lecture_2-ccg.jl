using BilevelJuMP

GRB_ENV = Gurobi.Env()

include("$(pwd())/instance_generation_cc.jl")

TotalFacilities = 10
TotalClients = 30
TotalScenarios = 50

instance = generate_instance(TotalFacilities, TotalClients, TotalScenarios)
expected_value_model = generate_full_problem(instance)
optimize!(expected_value_model)


function generate_full_minimax_problem(instance::Instance; solver=Gurobi)
    
    I, J, S, K, P, O, V, U, T, D, bigM = unroll_instance(instance)

    # Initialize model
    m = Model(solver.Optimizer)
    
    # Decision variables
    @variable(m, x[I], Bin)     # 1 if facility is located at i ∈ I, 0 otherwise.
    @variable(m, y[I] >= 0)     # Capacity decided for facility i ∈ I
    @variable(m, w[I,J,S] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(m, z[J,S] >= 0)   # Shortage in location j ∈ J in scenario s ∈ S
    @variable(m, θ >= 0)

    # Constraints
    # Maximum number of servers
    @constraint(m, numServers,
        sum(x[i] for i in I) <= K
    )
    
    # Capacity limits: cannot deliver more than capacity decided, 
    #   and only if facility was located
    @constraint(m, capBal[i in I, s in S],
        sum(w[i,j,s] for j in J) <=  y[i]
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
        sum(O[i] * x[i] + V[i] * y[i] for i in I) 
    )

    @constraint(m, Minimax[s in S], 
        θ >= sum(T[i,j] * w[i,j,s] for i in I, j in J) + sum(U * z[j,s] for j in J)
    )
    
    @objective(m, Min, FirstStage + θ)
    
    return m  # Return the generated model
end

minimax = generate_full_minimax_problem(instance::Instance)
optimize!(minimax)


## Comparing solutions

@show x_exp = Int.(round.(value.(expected_value_model[:x]).data))
@show y_exp = value.(expected_value_model[:y])
@show obj_exp = objective_value(expected_value_model)

@show x_minmax = Int.(round.(value.(minimax[:x]).data))
@show y_minmax = value.(minimax[:y])
@show obj_minmax = objective_value(minimax)


## Column and constraint generation
# Generates the main problem
function generate_main(instance)
    
    I, J, S, K, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    main = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(main)
    
    @variable(main, x[I], Bin)
    @variable(main, 0 <= y[I] <= bigM)
    @variable(main, θ)

    @constraint(main, sum(x[i] for i in I) <= K)
    
    @objective(main, Min, sum(O[i] * x[i] + V[i] * y[i] for i in I) + θ)

    return main  
end

# Adds the columns and constraints to the main problem
function add_columns_and_constraints(instance, main, d_bar, iter)   
    
    I, J, S, K, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    x = main[:x]
    y = main[:y]
    θ = main[:θ] 
    
    # Adding new variables to the main model. We are registering new variable sets at each iteration,
    # which requires the name of the variable to be passed as a Symbol.
    main[Symbol("w_$(iter)")] = @variable(main, [I,J], lower_bound = 0.0, base_name = "w_$(iter)")
    main[Symbol("z_$(iter)")] = @variable(main, [J], lower_bound = 0.0, base_name = "z_$(iter)")
    
    # For clarity only, we allocate to local variables 
    w = main[Symbol("w_$(iter)")]
    z = main[Symbol("z_$(iter)")]
    
    @constraint(main, 
        θ >= sum(T[i,j] * w[i,j] for i in I, j in J) + sum(U * z[j] for j in J) 
    )
    
    @constraint(main, [i in I],
        sum(w[i,j] for j in J) <=  y[i]
    )

    @constraint(main, [i in I], 
        sum(w[i,j] for j in J) <= x[i] * bigM
    )
    
    @constraint(main, [j in J],
        sum(w[i,j] for i in I) >= d_bar[j] - z[j]
    )
      
    return main
end  


# Solve the main problem
function solve_main(ins, main)
    optimize!(main)
    return value.(main[:x]), value.(main[:y]), value(main[:θ]), objective_value(main)    
end



# Define the CC subproblem. A couple of variants
function generate_and_solve_dualized_subproblem(instance, x_bar, y_bar, Γ)
    
   I, J, S, K, P, O, V, U, T, D, bigM, D_average, D_deviation = unroll_instance(instance)
    
   sub_dual = Model(() -> Gurobi.Optimizer(GRB_ENV))
   set_optimizer_attribute(sub_dual, "NonConvex", 2) # Call spatial BB solver

   set_silent(sub_dual)
    
   @variable(sub_dual, u[I] <= 0)
   @variable(sub_dual, q[I] <= 0)
   @variable(sub_dual, r[J] >= 0)
   @variable(sub_dual, 0 <= g[J] <= 1)
   @variable(sub_dual, d[J] >= 0)

   @constraint(sub_dual, [i in I, j in J], 
      u[i] + q[i] + r[j] <= T[i,j]
   )
   @constraint(sub_dual, [j in J], 
      r[j] <= U
   )

   # Uncertainty set
   @constraint(sub_dual, [j in J],
      d[j] == D_average[j] + g[j] * D_deviation[j]
   )

   @constraint(sub_dual, [j in J],
      sum(g[j] for j in J) <= Γ
   )

   @objective(sub_dual, Max,  
      sum(bigM * x_bar[i] * q[i] for i in I) +
      sum(y_bar[i] * u[i] for i in I) +
      sum(d[j] * r[j] for j in J) 
   )   
   
   optimize!(sub_dual)
    
   d_bar = value.(sub_dual[:d])                     
   opt_value = objective_value(sub_dual)
    
   return d_bar, opt_value
end


# Uses the result that g is always integer anyways and apply linearisation
function generate_and_solve_linearized_subproblem(instance, x_bar, y_bar, Γ)
    
    I, J, S, K, P, O, V, U, T, D, bigM, D_average, D_deviation = unroll_instance(instance)
     
    sub_dual = Model(()->Gurobi.Optimizer(GRB_ENV)) 
    set_silent(sub_dual)
     
    #%
    @variable(sub_dual, u[I] <= 0)
    @variable(sub_dual, q[I] <= 0)
    @variable(sub_dual, r[J] >= 0)
    @variable(sub_dual, g[J], Bin)
   #  @variable(sub_dual, d[J] >= 0)
    @variable(sub_dual, b[J] >= 0)
 
    @constraint(sub_dual, [i in I, j in J], 
       u[i] + q[i] + r[j] <= T[i,j]
    )
    @constraint(sub_dual, [j in J], 
       r[j] <= U
    )
 
    # Uncertainty set
   #  @constraint(sub_dual, [j in J],
   #     d[j] == D_average[j] + g[j] * D_deviation
   #  )
 
    @constraint(sub_dual, [j in J],
       sum(g[j] for j in J) <= Γ
    )

    # Linearisation constraints
    @constraint(sub_dual, [j in J],
       b[j] <= r[j]
    )
 
    @constraint(sub_dual, [j in J],
       b[j] <= U * g[j]
    )

    @objective(sub_dual, Max,  
       sum(bigM * x_bar[i] * q[i] for i in I) +
       sum(y_bar[i] * u[i] for i in I) +
       sum(D_average[j] * r[j] + D_deviation[j] * b[j] for j in J) 
    )   
    
    optimize!(sub_dual)
     
    d_bar = [D_average[j] + value(g[j]) * D_deviation[j] for j in J]                 
    opt_value = objective_value(sub_dual)
     
    return d_bar, opt_value
 end

# Bilevel JuMP version
function generate_and_solve_bilevel_subproblem(instance, x_bar, y_bar, Γ)
    
   I, J, S, K, P, O, V, U, T, D, bigM, D_average, D_deviation = unroll_instance(instance)
     
   sub_dual = BilevelModel(()->Gurobi.Optimizer(GRB_ENV), mode = BilevelJuMP.SOS1Mode()) 
   set_silent(sub_dual)
     
    #%
   @variable(Upper(sub_dual), d[J] >= 0)
   @variable(Upper(sub_dual), 0 <= g[J] <= 1)
   @variable(Lower(sub_dual), w[I,J] >= 0)
   @variable(Lower(sub_dual), z[J] >= 0)
 
   # Capacity limits: cannot deliver more than capacity decided, 
   #   and only if facility was located
   @constraint(Lower(sub_dual), capBal[i in I],
      sum(w[i,j] for j in J) <=  y_bar[i]
   )

   @constraint(Lower(sub_dual), capLoc[i in I], 
      sum(w[i,j] for j in J) <= x_bar[i] * bigM
   )
   
   # Demand balance: Demand of active clients must be fulfilled
   @constraint(Lower(sub_dual), demBal[j in J],
      sum(w[i,j] for i in I) >= d[j] - z[j]
   )

   # Uncertainty set
   @constraint(Upper(sub_dual), [j in J],
      d[j] == D_average[j] + g[j] * D_deviation[j]
   )

   @constraint(Upper(sub_dual), [j in J],
      sum(g[j] for j in J) <= Γ
   )
  
   @objective(Lower(sub_dual), Min, 
      sum(T[i,j] * w[i,j] for i in I, j in J) + sum(U * z[j] for j in J)
   ) 

   @objective(Upper(sub_dual), Max, 
      sum(T[i,j] * w[i,j] for i in I, j in J) + sum(U * z[j] for j in J)
   ) 
   
   #%
   optimize!(sub_dual)
   
   d_bar = [value(d[j]) for j in J]                 
   opt_value = objective_value(sub_dual)
   
   return d_bar, opt_value
 end


 function cc_decomposition(ins; max_iter = 100, Γ = 10, sub_method = :linear)
    k = 1
    ϵ = 0.005
    LB = -Inf
    UB = +Inf
    gap = +Inf
    x_bar = zeros(length(ins.I))
    y_bar = zeros(length(ins.J))
    
    println("\nStarting CCG decomposition...")
    start = time()    
    
    if sub_method == :linear
        d_bar, f_sub = generate_and_solve_linearized_subproblem(ins, x_bar, y_bar, Γ);
    elseif sub_method == :dual    
        d_bar, f_sub = generate_and_solve_dualized_subproblem(ins, x_bar, y_bar, Γ);
    elseif sub_method == :bilevel
        d_bar, f_sub = generate_and_solve_bilevel_subproblem(ins, x_bar, y_bar, Γ);
    else    
        error("Invalid subproblem solution method chosen.")
    end

    main = generate_main(ins)
    main = add_columns_and_constraints(ins, main, d_bar, k) 

    while k <= max_iter && gap > ϵ
        x_bar, y_bar, θ_bar, f_main = solve_main(ins, main);
        # d_bar, f_sub = generate_and_solve_dualized_subproblem(ins, x_bar, y_bar, Γ);
        # d_bar, f_sub = generate_and_solve_linearized_subproblem(ins, x_bar, y_bar, Γ);
        d_bar, f_sub = generate_and_solve_bilevel_subproblem(ins, x_bar, y_bar, Γ);
        LB = f_main
        UB = min(UB, f_main - θ_bar + f_sub)
        gap = abs((UB - LB) / UB)
        println("Iter $(k): UB: $(round(UB, digits=2)), LB: $(round(LB, digits=2)), gap: $(round(100*gap, digits=2))%")
        
        if gap <= ϵ
            stop = time()
            println("\nOptimal found.\n Objective value: $(round(UB, digits=2))\n Total time: $(round(stop-start, digits=2))s\n gap: $(round(100*gap, digits=2))%")
            return x_bar, y_bar, UB
        else    
            main = add_columns_and_constraints(ins, main, d_bar, k)
            k += 1
        end
    end
    println("Maximum number of iterations exceeded.")
end

@time x_bar, y_bar, UB = cc_decomposition(instance, max_iter = 5, Γ = 5, sub_method = :linear)
@time x_bar, y_bar, UB = cc_decomposition(instance, max_iter = 5, Γ = 5, sub_method = :dual)
@time x_bar, y_bar, UB = cc_decomposition(instance, max_iter = 5, Γ = 5, sub_method = :bilevel)

UB_k = []
for i in 1:10
    x_bar, y_bar, UB = cc_decomposition(instance, max_iter = 5, Γ = i,)
    push!(UB_k, UB)
end

plot(UB_k, 
    label = "obj. value",
    xlabel = "Γ")