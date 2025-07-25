module IbPDemo

# Write your package code here.

using Plots
using ProgressMeter

include("types.jl")
include("utils.jl")


struct SimWorld <: World
    time_step::Float64 # Time length that occur per sim step
    obs_proc::Process # Parameters of an Observable Processes. Ties to sample function.
end
SimWorld(; timestep::Float64=1.0, proc::Process) = SimWorld(timestep, proc)

function simulate(world::SimWorld, aplan::PathPlan; duration::Float64=1000.0)
    agent = Human(model=make_naive_gp())
    agent.plan = aplan
    a = Animation()
    let agent_x = [], agent_y = [], whale_x = [], whale_y = []
        @showprogress for i in 1:duration
            update_process!(world.obs_proc, world.time_step)
            append!(whale_x, world.obs_proc.c_loc[1])
            append!(whale_y, world.obs_proc.c_loc[2])
            let plan_complete=follow_path!(agent, world)
                if plan_complete
                    break
                end
                # sleep(world.time_step)
            end
            append!(agent_x, agent.loc[1])
            append!(agent_y, agent.loc[2])
            p1 = heatmap(agent.model.model)
            plt1 = plot(p1, xlimits=(0,30), ylimits=(0,30); obsv=true)
            plot!(agent_x, agent_y, lc=:green, lw=2, la=0.5, ls=:dot, label="agent")
            scatter!(whale_x, whale_y, mc=:blue, ms=4, label="whales")
            frame(a, plt1)
        end
        optimize!(agent.model.model)
        p1 = heatmap(agent.model.model)
        plt1 = plot(p1, xlimits=(0,30), ylimits=(0,30); obsv=true)
        plot!(agent_x, agent_y, lc=:green, lw=2, la=0.5, ls=:dot, label="agent")
        scatter!(whale_x, whale_y, mc=:blue, ms=4, label="whales")
        for i in 1:30
            frame(a, plt1)
        end
        gif(a, fps=10)
    end
end

function sim_world(world::SimWorld; duration::Float64=1000.0)
    a = Animation()
    let whale_x = [], whale_y = []
        @showprogress desc="Simulating whale movement... " for i in 1:duration
            update_process!(world.obs_proc, world.time_step)
            append!(whale_x, world.obs_proc.c_loc[1])
            append!(whale_y, world.obs_proc.c_loc[2])
            plt1 = scatter(whale_x, whale_y, mc=:blue, ms=4, label="whales",
                           xlimits=(0,30), ylimits=(0,30))
            frame(a, plt1)
        end
        plt1 = scatter(whale_x, whale_y, mc=:blue, ms=4, label="whales",
                       xlimits=(0,30), ylimits=(0,30))
        for i in 1:30
            frame(a, plt1)
        end
        gif(a, fps=10)
    end
end

function sim_simple()
    let ws=WhaleSurfacings(scale=0.5, sites=[[10. 10.]; [20. 15.]; [25. 30.]], c_loc=[8.5 9.5]), sw=SimWorld(proc=ws)
        # agentp = PathPlan([[10. 10.];
        #                    [20. 20.];
        #                    [30. 30.];
        #                    [20. 30.];
        #                    [10. 20.];
        #                    [10. 10.];
        #                    [20. 10.]], 1)
        # agentp = PathPlan([[30. 0.];
        #                    [30. 10.];
        #                    [0. 10.];
        #                    [0. 20.];
        #                    [30. 20.];
        #                    [30. 30.];
        #                    [0. 30.];], 1)
        agentp = PathPlan([[10. 10.];
                           [20. 10.];
                           [20. 20.];
                           [30. 20.];
                           [30. 30.];
                           [30. 20.]], 1)
        simulate(sw, agentp)
    end
end

function sim_whale()
    let ws=WhaleSurfacings(scale=0.3, sites=[[10. 10.]; [25. 15.]; [30. 30.]], c_loc=[8.5 9.5]), sw=SimWorld(proc=ws)
        sim_world(sw)
    end
end

export SimWorld, simulate, WhaleSurfacings, sim_simple, sim_whale

end