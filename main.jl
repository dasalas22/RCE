using FileIO, JLD2, BilevelJuMP, Clarabel, JuMP, Gurobi, Printf, JSON

include("polytopesampling.jl")

function checkBounded(A,B,b,bigM) #Check if polyhedron is bounded, otherwise add bigM as bound

    m = size(A,1);
    nx = size(A,2);
    ny = size(B,2);

    xbox = Inf*ones(nx,2)

    model = direct_model(Gurobi.Optimizer())
    set_silent(model)
    @variable(model, x[1:nx]);
    @variable(model, y[1:ny]);
    
    #the next to lines are to force vectors to be interpreted matrices
    A = reshape(A,m,nx)
    B = reshape(B,m,ny)

    @constraint(model, A*x + B*y .<=b);
    
    for i = 1:nx
        #Upper bound
        @objective(model, Max, x[i]);
        optimize!(model)
        status = termination_status(model)
        if status != OPTIMAL
            newrowA = zeros(1,nx)
            newrowA[i] = 1
            newrowB = zeros(1,ny)

            A = vcat(A,newrowA)
            B = vcat(B,newrowB)
            b = vcat(b,bigM)

            xbox[i,2] = bigM
        else
            xbox[i,2] = objective_value(model)
        end
        
        #Lower bound
        @objective(model, Min, x[i]);
        optimize!(model)
        status = termination_status(model)
        if status != OPTIMAL
            newrowA = zeros(1,nx)
            newrowA[i] = -1
            newrowB = zeros(1,ny)

            A = vcat(A,newrowA)
            B = vcat(B,newrowB)
            b = vcat(b,bigM)

            xbox[i,1] = -bigM
        else
            xbox[i,1] = objective_value(model)
        end
    end

    for i = 1:ny
        #Upper bound
        @objective(model, Max, y[i]);
        optimize!(model)
        status = termination_status(model)
        if status != OPTIMAL
            newrowA = zeros(1,nx)
            newrowB = zeros(1,ny)
            newrowB[i] = 1

            A = vcat(A,newrowA)
            B = vcat(B,newrowB)
            b = vcat(b,bigM)
        end
        #Lower bound
        @objective(model, Min, y[i]);
        optimize!(model)
        status = termination_status(model)
        if status != OPTIMAL
            newrowA = zeros(1,nx)
            newrowB = zeros(1,ny)
            newrowB[i] = -1

            A = vcat(A,newrowA)
            B = vcat(B,newrowB)
            b = vcat(b,bigM)
        end
    end
    return A,B,b,xbox
end

function createBaseFollower(x,A,B,b,fx,fy) # Create follower problem for a given leader vector x
    m = size(A,1);
    nx = size(A,2);
    ny = size(B,2);

    #model = Model(Clarabel.Optimizer) #We (try to) ensure that we get a strictly complementary solution
    model = direct_model(Gurobi.Optimizer()) #Gurobi seems to be more stable that Clarabel numerically
    set_silent(model)
    set_optimizer_attribute(model, "Method", 2) #We (try to) ensure that we get a strictly complementary solution
    set_optimizer_attribute(model, "Threads", 1)

    @variable(model, y[1:ny]);
    
    #next to lines are to force vectors to be interpreted matrices
    A = reshape(A,m,nx)
    B = reshape(B,m,ny)

    @constraint(model, cons, B*y .<= b - A*x);

    @objective(model, Min, sum(fy.*y) );
    
    return model
end

function updateAndSolveFollower(model,x,A,B,b,fx,fy) # Create follower problem for a given leader vector x
    m = size(A,1);
    nx = size(A,2);
    ny = size(B,2);

    newrhs =  b - A*x

    for i=1:m
        set_normalized_rhs(model[:cons][i], newrhs[i])
    end
   
    optimize!(model)
    #println("Solve time follower ", solve_time(model))
    status = termination_status(model)
    if status != OPTIMAL
        return NaN, NaN, status
    end
    ysol = JuMP.value.(model[:y])

    duals = zeros(m,1)
    for i = 1:m
        duals[i] = JuMP.dual(model[:cons][i])
    end
    
    return ysol, duals, status
end

function updateAndSolveLeader(model, A, B, b, activeconstrs)
    
    nrows = size(A,1);
    nx = size(A,2);
    ny = size(B,2);
    
    A = reshape(A,nrows,nx)
    B = reshape(B,nrows,ny)
    
    activeidx = findall(a->abs(a)>1E-8, vec(activeconstrs))
    nactiveidx = length(activeidx)

    #delete(model, model[:binding])
    #unregister(model, :binding)

    binding = @constraint(model,
        [j=1:nactiveidx], sum( A[activeidx[j],k]*model[:x][k] for k=1:nx) + sum(B[activeidx[j],k]*model[:y][k] for k=1:ny) >= b[activeidx[j]]
    );
    
    optimize!(model)
    status = termination_status(model)
    #println("Solve time leader ", solve_time(model))
    if status != OPTIMAL
        println("Error in leader problem ", status)
        # write_to_file(model, "infeas.lp")
        for j=1:nactiveidx
            delete(model, binding[j])
        end
        return NaN,Inf, status
    end
    
    xval = JuMP.value.(model[:x])
    yval = JuMP.value.(model[:y])
    objval = objective_value(model)

    for j=1:nactiveidx
        delete(model, binding[j])
    end
    
    return xval, yval, objval, status

end

function formulateBaseLeader(A, B, b, Fx, Fy)
    
    nrows = size(A,1);
    nx = size(A,2);
    ny = size(B,2);

    model = direct_model(Gurobi.Optimizer())
    set_silent(model)
    set_optimizer_attribute(model, "Threads", 1)
    @variable(model, x[1:nx]);
    @variable(model, y[1:ny]); 
    
    A = reshape(A,nrows,nx)
    B = reshape(B,nrows,ny)

    @constraints(model, begin
        [j=1:nrows], sum( A[j,k]*x[k] for k=1:nx) + sum(B[j,k]*y[k] for k=1:ny) <= b[j]
    end);
        
    @objective(model, Min, sum(Fx.*x)+sum(Fy.*y));

    return model

end

function samplevector_box(nx, xbox)
    v = zeros(nx,1) 

    v = xbox[:,1] + (xbox[:,2] - xbox[:,1]).*rand(nx)
    return v
end

# Variables mostly used for reports
dim = 0
ncons = 0

couplingtolower = true
if ARGS[1] == "knapsack"
    name = "ContinuousKnapsack_instance.jl"
    couplingtolower = false
else
    name = ARGS[1]
end

println("\n========\n Running ", name, "\n=========\n")

include(@sprintf("%s",name))

(ncoupling,) = size(Gx)
if couplingtolower && ncoupling >= 1
    A = vcat(Gx,gx)
    B = vcat(Gy,gy)
    b = vcat(-bG,-bg)
else
    A = gx
    B = gy
    b = -bg
end

A,B,b,xbox = checkBounded(A,B,b,1000) #We add artificial bounds if we need to
nx = size(A,2);
ny = size(B,2);
m = size(A,1);

dim = nx+ny;
ncons = m;


############################
# Check if problem is easy #
############################
println("\n========\n Bilevel solver plain \n=========\n")

modelBilevelPlain = BilevelModel()
set_optimizer(modelBilevelPlain,Gurobi.Optimizer)
if ARGS[3] == "SOS1"
    BilevelJuMP.set_mode(modelBilevelPlain, BilevelJuMP.SOS1Mode())
else
    BilevelJuMP.set_mode(modelBilevelPlain, BilevelJuMP.IndicatorMode())
end

set_optimizer_attribute(modelBilevelPlain, "Threads", 1)
set_optimizer_attribute(modelBilevelPlain, "TimeLimit", 3600)

@variable(Upper(modelBilevelPlain), xbilPlain[1:nx])
@variable(Lower(modelBilevelPlain), ybilPlain[1:ny])

@objective(Upper(modelBilevelPlain), Min, sum(Fx.*xbilPlain)+sum(Fy.*ybilPlain))

@objective(Lower(modelBilevelPlain), Min, sum(fx.*xbilPlain)+sum(fy.*ybilPlain))
@constraints(Lower(modelBilevelPlain), begin
[j in 1:m], sum( A[j,k]*xbilPlain[k] for k=1:nx) + sum(B[j,k]*ybilPlain[k] for k=1:ny) <= b[j]
end)

optimize!(modelBilevelPlain)

realoptPlain = objective_value(modelBilevelPlain)
if solve_time(modelBilevelPlain) < 5
    println("SUMMARY: ", name, " Gurobi took < 5 secs to solve")
    #exit(0)
end
############################

#### Stochastic method ####
runStoch = true
Nsamplesx = parse(Int,ARGS[2])
bestx = zeros(nx,1)
bestxval = Inf
besty = zeros(ny,1)

bestx_sampled = zeros(nx,1)
bestxval_sampled = Inf
besty_sampled = zeros(ny,1)

seenlabels = Set()
labelcounts = zeros(Int64,0)
labelcounts50 = zeros(Int64,0)
labelcounts100 = zeros(Int64,0)
labelcounts200 = zeros(Int64,0)

labelid = Dict() 
success_count = 0
fail_count = 0
max_fail_count = 2*Nsamplesx

if runStoch
    
    modelLead  = formulateBaseLeader(A, B, b, Fx, Fy)
    modelFollower = createBaseFollower(bestx,A,B,b,fx,fy)
    
    #write_to_file(modelLead,"leadermodel.lp")
    start = time()

    samples_x = uniform_sample_x(A, B, b, Nsamplesx)
    samplecount = 1

    time_sample_gen = time() - start
    
    #println("INFO: Time to generate samples ", time() - start)
    start = time()

    while success_count < Nsamplesx && fail_count <= max_fail_count
        #x_fix = samplevector_box(nx, xbox)
        x_fix = samples_x[1:nx, samplecount]
        global samplecount += 1

        ysol,activeconstrs, status = updateAndSolveFollower(modelFollower,x_fix,A,B,b,fx,fy)

        if status != OPTIMAL
            #println("Follower status ", status)
            global fail_count += 1
            println("SUMMARY: ", name, " Warning, samples should always be feasible")
            if fail_count > max_fail_count #this means we didn't suceed in sampling any feasible solution
                println("SUMMARY: ", name, " No feasible sample generated")
                exit(0)
            end

            continue
        end

        global success_count += 1

        value_sampled = sum(Fx.*x_fix)+sum(Fy.*ysol)
        if value_sampled < bestxval_sampled
            global bestxval_sampled = value_sampled
            global bestx_sampled = x_fix
            global besty_sampled = ysol
        end

        Ind = findall(a->abs(a)>1E-5, vec(activeconstrs));
        currentlabel = Set(Ind)
        if currentlabel in seenlabels
            
            labelcounts[labelid[currentlabel]] += 1
            if success_count <= 50
                labelcounts50[labelid[currentlabel]] += 1
            elseif success_count <= 100
                labelcounts100[labelid[currentlabel]] += 1
            else
                labelcounts200[labelid[currentlabel]] += 1
            end
            #println("Skipped")
            continue #in this case we skip the optimization for the leader
        else
            push!(seenlabels,currentlabel)
            labelid[currentlabel] = length(seenlabels)
            append!( labelcounts, 1 )
            append!( labelcounts50, (success_count <= 50 ) ? 1 : 0 )
            append!( labelcounts100, (success_count > 50 && success_count <= 100 ) ? 1 : 0 )
            append!( labelcounts200, (success_count > 100 ) ? 1 : 0 )
        end

        xopt, yopt, optval, status = updateAndSolveLeader(modelLead, A, B, b, activeconstrs)

        if optval < bestxval
            global bestxval = optval
            global bestx = xopt
            global besty = yopt
        end
    end

    timestoch = time() - start

    
end


println("\n========\n Bilevel solver WS-Cell \n=========\n")

modelBilevel = BilevelModel()
set_optimizer(modelBilevel,Gurobi.Optimizer)
if ARGS[3] == "SOS1"
    BilevelJuMP.set_mode(modelBilevel, BilevelJuMP.SOS1Mode())
else
    BilevelJuMP.set_mode(modelBilevel, BilevelJuMP.IndicatorMode())
end

set_optimizer_attribute(modelBilevel, "Threads", 1)
set_optimizer_attribute(modelBilevel, "TimeLimit", 3600)

@variable(Upper(modelBilevel), xbil[1:nx])
@variable(Lower(modelBilevel), ybil[1:ny])

@objective(Upper(modelBilevel), Min, sum(Fx.*xbil)+sum(Fy.*ybil))

@objective(Lower(modelBilevel), Min, sum(fx.*xbil)+sum(fy.*ybil))
@constraints(Lower(modelBilevel), begin
[j in 1:m], sum( A[j,k]*xbil[k] for k=1:nx) + sum(B[j,k]*ybil[k] for k=1:ny) <= b[j]
end)

for i = 1:nx
    set_start_value(xbil[i], bestx[i])
end
for i = 1:ny
    set_start_value(ybil[i], besty[i])
end

optimize!(modelBilevel)

realopt = objective_value(modelBilevel)


println("\n========\n Bilevel solver WS-Sample \n=========\n")

modelBilevel_sample = BilevelModel()
set_optimizer(modelBilevel_sample,Gurobi.Optimizer)
if ARGS[3] == "SOS1"
    BilevelJuMP.set_mode(modelBilevel_sample, BilevelJuMP.SOS1Mode())
else
    BilevelJuMP.set_mode(modelBilevel_sample, BilevelJuMP.IndicatorMode())
end

set_optimizer_attribute(modelBilevel_sample, "Threads", 1)
set_optimizer_attribute(modelBilevel_sample, "TimeLimit", 3600)

@variable(Upper(modelBilevel_sample), xbil[1:nx])
@variable(Lower(modelBilevel_sample), ybil[1:ny])

@objective(Upper(modelBilevel_sample), Min, sum(Fx.*xbil)+sum(Fy.*ybil))

@objective(Lower(modelBilevel_sample), Min, sum(fx.*xbil)+sum(fy.*ybil))
@constraints(Lower(modelBilevel_sample), begin
[j in 1:m], sum( A[j,k]*xbil[k] for k=1:nx) + sum(B[j,k]*ybil[k] for k=1:ny) <= b[j]
end)

for i = 1:nx
    set_start_value(xbil[i], bestx_sampled[i])
end
for i = 1:ny
    set_start_value(ybil[i], besty_sampled[i])
end

optimize!(modelBilevel_sample)

realopt_sampled = objective_value(modelBilevel_sample)


println("\n========\n Summary \n=========\n")

println("SUMMARY-Short: ", name, " ObjRatios ", realopt_sampled/realoptPlain, " ", realopt/realoptPlain, " TimeRatios ", solve_time(modelBilevel_sample)/solve_time(modelBilevelPlain), " ", solve_time(modelBilevel)/solve_time(modelBilevelPlain), " SamplingTime ", time_sample_gen, " TimeCell ", timestoch )

# Assuming the variables 'name', 'bestxval', 'bestxval_sampled', 'timestoch', etc. are already defined

# Create a dictionary to hold the information
summary_dict = Dict(
    "name" => name,
    "best_val_cell_search" => bestxval,
    "best_val_sampled" => bestxval_sampled,
    "gap_cell_search" => (bestxval - realoptPlain)/abs(realoptPlain),
    "gap_sampled" => (bestxval_sampled - realoptPlain)/abs(realoptPlain),
    "time_cell" => timestoch,
    "time_sample_gen" => time_sample_gen,
    "exact_opt_GRB_plain" => realoptPlain,
    "time_exact_GRB_plain" => solve_time(modelBilevelPlain),
    "exact_opt_ws_random" => realopt_sampled,
    "time_exact_ws_random" => solve_time(modelBilevel_sample),
    "exact_opt_ws_cell" => realopt,
    "time_exact_ws_cell" => solve_time(modelBilevel),
    "n_seen_labels" => length(seenlabels),
    "label_counts" => labelcounts,
    "label_counts50" => labelcounts50,
    "label_counts100" => labelcounts100,
    "label_counts200" => labelcounts200,
    "success_count" => success_count,
    "fail_count" => fail_count,
    "obj_ratios" => [
        realopt_sampled/realoptPlain,
        realopt/realoptPlain
    ],
    "time_ratios" => [
        solve_time(modelBilevel_sample)/solve_time(modelBilevelPlain),
        solve_time(modelBilevel)/solve_time(modelBilevelPlain)
    ]
)

# Save the dictionary to a JSON file
json_file = name * "." *ARGS[2]* ".out.json"
open(json_file, "w") do io
    JSON.print(io, summary_dict, 4)
end

println("Summary saved to $json_file")
