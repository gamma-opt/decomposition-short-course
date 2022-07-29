using Gurobi
const GRB_ENV = Gurobi.Env()
myGurobi = ()->Gurobi.Optimizer(GRB_ENV)

using JuMP
using Plots
using LinearAlgebra
using Random

Random.seed!(1234)

struct Instance
    # sets
    I  # Set of facilities
    J  # Set of clients
    S  # Set of scenarios
    # Parameters 
    N  # Maximum number of facilities
    P  # Probabilities of scenarios s ∈ S 
    O  # Cost of opening facility at i ∈ I
    V  # Variable capacity cost  
    U  # Cost of unmet demand at j ∈ J
    T  # Transportation cost from i ∈ I to j ∈ J
    D  # Demand in location j ∈ J
    bigM  # BigM for capacity constraint
    D_average   # average demand
    D_deviation # max deviation
    loc_i # Coordinates of facilities i ∈ I
    loc_j # Coordinates of clients j ∈ J
end

function generate_instance(TotalServers, TotalClients, TotalScenarios)
    I = 1:TotalServers
    J = 1:TotalClients
    S = 1:TotalScenarios

    # Parameters
    N = ceil(0.5 * TotalServers)           # Half (rounded up) of all servers can be used
    P = [1/TotalScenarios for s in S]      # All scenarios have equal probability
    O = [rand(40:80) for i in I]           # Fixed cost for locating facility
    V = [rand(1:10) for i in I]            # Variable capacity cost   
    U = 500                                # High cost for unmet demand
    loc_i = [(rand(), rand()) for i in I]  # Random 2D-coordinates for clients (coordinates are represented by a tuple)
    loc_j = [(rand(), rand()) for j in J]  # Random 2D-coordinates for servers
    
    # The transportation cost is the Euclidean distance between the server i and client j
    T = ceil.(10 .* [sqrt((loc_i[i][1]-loc_j[j][1])^2+(loc_i[i][2]-loc_j[j][2])^2) for i in I, j in J])
    
    D_average = [rand(10:50) for j in J]   # Random demand average for each client           
    D_deviation = [Int(ceil(0.5 * D_average[j])) for j in J]
    D = [max(ceil(D_average[j] + rand(-D_deviation[j]:D_deviation[j])), 0) for j in J, s in S] # Random demand deviation per scenario
    max_D, index = findmax(D, dims=2)  # finds maximum among columns
    bigM = sum(max_D)                  # capacity big M

    return Instance(I, J, S, N, P, O, V, U, T, D, bigM, D_average, D_deviation, loc_i, loc_j)
end  

function unroll_instance(instance::Instance)
    I = instance.I 
    J = instance.J
    S = instance.S
    N = instance.N
    P = instance.P
    O = instance.O
    V = instance.V
    U = instance.U
    T = instance.T
    D = instance.D
    bigM = instance.bigM
    D_average = instance.D_average
    D_deviation = instance.D_deviation

    return I, J, S, N, P, O, V, U, T, D, bigM, D_average, D_deviation
end

function generate_full_problem(instance::Instance)
    
    I, J, S, N, P, O, V, U, T, D, bigM = unroll_instance(instance)

    # Initialize model
    m = Model(myGurobi)
    
    # Decision variables
    @variable(m, x[I], Bin)     # 1 if facility is located at i ∈ I, 0 otherwise.
    @variable(m, y[I] >= 0)     # Capacity decided for facility i ∈ I
    @variable(m, w[I,J,S] >= 0) # Flow between facility i ∈ I and client j ∈ J in scenario s ∈ S
    @variable(m, z[J,S] >= 0)   # Shortage in location j ∈ J in scenario s ∈ S

    # Constraints
    # Maximum number of servers
    @constraint(m, numServers,
        sum(x[i] for i in I) <= N
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

    SecondStage = @expression(m, 
        sum(P[s] * (
                sum(T[i,j] * w[i,j,s] for i in I, j in J)
                + sum(U * z[j,s] for j in J)) 
        for s in S)
    )
    
    @objective(m, Min, FirstStage + SecondStage)
    
    return m  # Return the generated model
end

