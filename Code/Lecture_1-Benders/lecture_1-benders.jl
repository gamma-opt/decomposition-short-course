## Lecture 1 - Benders decomposition

# Include random instance generation and solving the 2SSP
include("$(pwd())/instance_generation.jl")

# Generating an instance and solving full space for checking
TotalFacilities = 10
TotalClients = 50
TotalScenarios = 100

# Solving the full space problem for reference
instance = generate_instance(TotalFacilities, TotalClients, TotalScenarios)
fullmodel = generate_full_problem(instance)
optimize!(fullmodel)


## Benders decomposition

# Generates the main problem
function generate_main(instance)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    main = Model(myGurobi)
    set_silent(main)
    
    @variable(main, x[I], Bin)
    @variable(main, y[I] >= 0)
    @variable(main, θ)

    @constraint(main, sum(x[i] for i in I) <= N)
    @constraint(main, sum(y[i] for i in I) <= bigM) # This is to guarantee boundedness

    @objective(main, Min, sum(O[i] * x[i] + V[i] * y[i] for i in I) + θ)

    return main  
end

# Solve the main problem
function solve_main(main)
    optimize!(main)
    return value.(main[:x]), value.(main[:y]), value(main[:θ]), objective_value(main)    
end

#= Generate and solve the primal subproblem for a given x_bar. 
# For test purposes only; if the dual is correct, the objective value of
the dual subproblem must be the same as this. =#
function generate_and_solve_primal_subproblem(instance, x_bar, y_bar)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    sub = Model(myGurobi)
    set_silent(sub)
    
    @variable(sub, w[I,J,S] >= 0)
    @variable(sub, z[J,S] >= 0)

    @constraint(sub, capBal[i in I, s in S],
        sum(w[i,j,s] for j in J) <=  y_bar[i]
    )

    @constraint(sub, capLoc[i in I, s in S], 
        sum(w[i,j,s] for j in J) <= x_bar[i] * bigM
    )
    
    @constraint(sub, demBal[j in J, s in S],
        sum(w[i,j,s] for i in I) >= D[j,s] - z[j,s]
    )

    @objective(sub, Min,  
        sum(P[s] * (
            sum(T[i,j] * w[i,j,s] for i in I, j in J)
            + sum(U * z[j,s] for j in J)
        ) for s in S)
    )
    
    optimize!(sub)
    return objective_value(sub)
    
end

# The dual version of the subproblem
function generate_and_solve_dual_subproblem(instance, x_bar, y_bar)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
     
    sub_dual = Model(myGurobi)
    set_silent(sub_dual)
     
    @variable(sub_dual, u[I,S] <= 0)
    @variable(sub_dual, q[I,S] <= 0)
    @variable(sub_dual, r[J,S] >= 0)
 
    @constraint(sub_dual, [i in I, j in J, s in S], 
       u[i,s] + q[i,s] + r[j,s] <= P[s] * T[i,j]
    )
    @constraint(sub_dual, [j in J, s in S], 
       r[j,s] <= P[s] * U
    )
    @objective(sub_dual, Max,  
       sum( 
          sum(bigM * x_bar[i] * q[i,s] for i in I) +
          sum(y_bar[i] * u[i,s] for i in I) +
          sum(D[j,s] * r[j,s] for j in J) 
       for s in S)   
    )
 
    optimize!(sub_dual)
     
    u_bar = value.(sub_dual[:u])
    q_bar = value.(sub_dual[:q])                     
    r_bar = value.(sub_dual[:r])                     
    opt_value = objective_value(sub_dual)
     
    return u_bar, q_bar, r_bar, opt_value
 end

# Test that the primal and dual solutions are the same
@show x_bar = Int.(round.(value.(fullmodel[:x]).data))
@show y_bar = value.(fullmodel[:y])

u_bar, q_bar, r_bar, dual_obj = generate_and_solve_dual_subproblem(instance, x_bar, y_bar)
primal_obj = generate_and_solve_primal_subproblem(instance, x_bar, y_bar)

# This throws an error if they do not match
@assert(primal_obj ≈ dual_obj) 


# Add the Benders cut, given current dual values
function add_benders_cut(instance, main, u_bar, q_bar, r_bar)   
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)
    
    x = main[:x]
    y = main[:y]
    θ = main[:θ]

    @constraint(main, 
        θ >= sum( 
                sum(bigM * x[i] * q_bar[i,s] for i in I) +
                sum(y[i] * u_bar[i,s] for i in I) +
                sum(D[j,s] * r_bar[j,s] for j in J) 
            for s in S) 
    )

    return main
end  


# The main function code
function benders_decomposition(ins; max_iter = 100)
    k = 1
    ϵ = 1e-4
    LB = -Inf
    UB = +Inf
    gap = +Inf
    x_bar = zeros(length(ins.J))
    y_bar = zeros(length(ins.J))
    
    start = time()    
    println("\nStarting Benders decomposition...\n")
    u_bar, q_bar, r_bar, f_sub = generate_and_solve_dual_subproblem(ins, x_bar, y_bar);
    main = generate_main(ins)
    main = add_benders_cut(ins, main, u_bar, q_bar, r_bar) 

    while k <= max_iter && gap > ϵ
        x_bar, y_bar, θ_bar, f_main = solve_main(main);
        u_bar, q_bar, r_bar, f_sub = generate_and_solve_dual_subproblem(ins, x_bar, y_bar);

        LB = f_main
        UB = min(UB, f_main - θ_bar + f_sub)
        gap = abs((UB - LB) / UB)
        println("Iter $(k): UB: $(round(UB, digits=2)), LB: $(round(LB, digits=2)), gap $(round(100.0*gap, digits=4))%")
        
        if gap <= ϵ
            stop = time()
            println("\nOptimal found. \nObjective value: $(round(UB, digits=2)) \n Total time: $(round(stop-start, digits=2))s \n gap: $(round(100.0*gap, digits=4))%\n")
            return
        else    
            main = add_benders_cut(ins, main, u_bar, q_bar, r_bar)  
            k += 1
        end
    end
    println("Maximum number of iterations exceeded")
end

benders_decomposition(instance, max_iter = 100)
@show obj = objective_value(fullmodel);