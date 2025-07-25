using LinearAlgebra

abstract type Process end

mutable struct WhaleSurfacings <: Process
    n::Float64 # Radius of whale surfacing visibility (meters)
    sc::Float64 # Time scale of whale movement (for simplicity, m/s shift of process)
    cores::Matrix{Float64} # Core sites of whale sightings
    t_site::Integer # Index of current target site of whales
    c_loc::Matrix{Float64} # Current location of whales
end
WhaleSurfacings(; n=30., scale=0.5, sites, start_site=1, c_loc) = WhaleSurfacings(n, scale, sites, start_site, c_loc)
get_target(ws::WhaleSurfacings) = reshape(ws.cores[ws.t_site,:],1,:)

function update_process!(ws::WhaleSurfacings, timestep::Float64)
    let sc=ws.sc, ts=timestep, target=get_target(ws), loc=ws.c_loc, d=sc*ts, diff_vc=target-loc, diff_rt=d/norm(diff_vc)
        # print("Updating process...\nLocation: ", loc, " | Target: ", target)
        ws.c_loc = diff_rt.*diff_vc .+ loc
        if diff_rt >= 1
            ws.t_site = mod(ws.t_site, size(ws.cores)[1])+1
        end
    end
end
sample(proc::WhaleSurfacings, loc::Matrix{Float64}) = norm(proc.c_loc - loc) < proc.n ? (proc.n - norm(proc.c_loc - loc)) / proc.n : 0.01

function follow_path!(agent::Agent, world::World)
    let act=get_action(agent.plan), target=get_target(agent.plan), wpt_achieved=goto_waypoint!(agent, world, target)
        take_observation!(agent, world)
        if wpt_achieved
            # println("Achieved waypoint: ", target)
            # take_observation!(agent, world)
            if act+1>get_plan_length(agent.plan)
                return true
            end
            set_target(agent.plan, act+1)
        end
    end
    return false
end

""" Go towards the specified waypoint for one world timestep.

If overshot, then remain at waypoint; return True.
If not arrived, then move for entire timestep; return False.
"""
function goto_waypoint!(agent::Agent, world::World, wpt::Matrix{Float64})
    let rv = wpt-agent.loc, r = norm(rv), d = world.time_step*agent.speed, dor=d/r
        if dor>=1-eps()
            agent.loc = wpt
            return true
        else
            agent.loc = [(dor*rv[1]+agent.loc[1]) (dor*rv[2]+agent.loc[2])]
            return false
        end
    end
end

function take_observation!(agent::Agent, world::World)
    let data = sample(world.obs_proc, agent.loc)
        # println("Observed: ", data)
        agent.model = GPModel(update_model(agent.model, agent.loc, data))
    end
end