using LinearAlgebra
using Random
using Convex
using Gurobi, JuMP

function checkImpliedInequalities(A,b) 

    fulldim_tol = 1e-6

    m = size(A,1);
    n = size(A,2);

    isfulldim = true
    impliedeq = zeros(m)

    model = direct_model(Gurobi.Optimizer())
    set_silent(model)
    @variable(model, x[1:n]);
    
    #next to lines are to force vectors to be interpreted matrices
    A = reshape(A,m,n)

    @constraint(model, A*x .<=b);
    
    for i = 1:m
        @objective(model, Min, sum(A[i,j]*x[j] for j=1:n) );
        optimize!(model)
        status = termination_status(model)
        if status == OPTIMAL && objective_value(model) >= b[i] - fulldim_tol
            if isfulldim
                println("INFO: Polytope is not fulldimensional")
            end
            isfulldim = false
            impliedeq[i] = 1
        end
    end
    return isfulldim,impliedeq
end

# Hit-and-Run sampling method
function hit_and_run_sampling(A, B, b, x0, y0, num_samples)
   
    nx = size(A, 2)
    ny = size(B, 2)

    m = size(A, 1)

    samples = zeros(nx, num_samples)
    samples[:, 1] = x0

    model = direct_model(Gurobi.Optimizer())
    set_silent(model)
    @variable(model, t);
    @variable(model, y[1:ny]);

    # Random direction
    d = randn(nx)
    d /= norm(d)

    #@constraint(model, con, A*(samples[:, 1] + t*d) + B*y <=b);
    @constraint(model, con[j in 1:m], sum(A[j,k]*(samples[k, 1] + t*d[k]) for k=1:nx)  + sum(B[j,k]*y[k] for k=1:ny) <=b[j]);

    for i in 2:num_samples

        # Calculate t_min and t_max
        t_min = -Inf
        t_max = Inf

        @objective(model, Min, t);
        optimize!(model)
        status = termination_status(model)

        if status == OPTIMAL
            t_min = value.(t)
        else
            error("Sampling error when finding tmin.")
        end

        @objective(model, Max, t);
        optimize!(model)
        status = termination_status(model)

        if status == OPTIMAL
            t_max = value.(t)
        else
            error("Sampling error when finding tmax.")
        end

        tstep = t_min + (t_max - t_min) * rand()

        samples[:, i] = samples[:, i-1] + tstep * d

        # New random direction
        d = randn(nx)
        d /= norm(d)

        newcoef = A*d
        newrhs = b - A*samples[:, i]
        for j in 1:m
            set_normalized_coefficient(con[j], t, newcoef[j])
            set_normalized_rhs(con[j], newrhs[j])
        end
        #@constraint(model, con, A*(samples[:, i-1] + t*d) + B*y .<=b);
    end

    return samples
end

# Uniform sampling within the polytope
function uniform_sample_x(A, B, b, num_samples)

    m = size(A,1);
    nx = size(A,2);
    ny = size(B,2);

    samples = zeros(nx, num_samples)

    isfulldim = true
    impliedeq = zeros(m)
    #skipping full dim check
    #isfulldim,impliedeq = checkImpliedInequalities(hcat(A,B),b) 

    Aineq = A[findall(impliedeq .== 0), :]  # Select rows where impliedeq[i] == 0
    Bineq = B[findall(impliedeq .== 0), :]
    bineq = b[findall(impliedeq .== 0)]     # Select corresponding entries from b
    Aeq = A[findall(impliedeq .== 1), :]    # Select rows where impliedeq[i] == 1
    Beq = B[findall(impliedeq .== 1), :]
    beq = b[findall(impliedeq .== 1)]       # Select corresponding entries from b
        
    Nx = I
    xyp = zeros(nx+ny,1)

    if !isfulldim
        N = nullspace(hcat(Aeq,Beq))
        Nx = N[:,1:nx]
        xyp = pinv(hcat(Aeq,Beq))*beq

        Aproj = hcat(Aineq,Bineq)*N[:,1:nx]
        Bproj = hcat(Aineq,Bineq)*N[:,(nx+1):(nx+ny)]
        bproj = bineq - hcat(Aineq,Bineq)*xyp

    else
        Aproj = Aineq
        Bproj = Bineq
        bproj = bineq
    end
    
    x0,y0 = find_starting_point(Aproj, Bproj, bproj)
    # Hit-and-Run for sampling
    samplesprojx = hit_and_run_sampling(Aproj, Bproj, bproj, x0, y0, num_samples)

    for i in 1:num_samples
        samples[:, i] = Nx*samplesprojx[:,i] + xyp[1:nx]
    end

    return samples
end

# Find a point inside the polytope using Linear Programming
function find_starting_point(A, B, b)
    nx = size(A, 2)
    ny = size(B, 2)

    model = direct_model(Gurobi.Optimizer())
    @variable(model, r >= 0)  # Radius of the Chebyshev ball
    @variable(model, x[1:nx])  # Center of the Chebyshev ball
    @variable(model, y[1:ny])
    
    # Constraints: A * x + r * ||a_i|| <= b_i for all i
    for i in 1:size(A, 1)
        @constraint(model, dot(A[i, :], x) + dot(B[i, :], y) + r * norm(vcat(A[i, :], B[i, :])) <= b[i])
    end
    
    # Maximize the radius
    @objective(model, Max, r)
    
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return value.(x), value.(y)
    else
        error("Failed to find the Chebyshev center.")
    end
end
