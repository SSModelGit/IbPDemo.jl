using GaussianProcesses
using Distributions

abstract type World end

abstract type Model end

struct GPModel <: Model
    model::GPE
end
make_naive_gp(dim::Integer=2) = GPModel(GPE(Matrix{Float64}(undef,dim,0),Float64[],MeanZero(),SE(zeros(dim),0.0)))
update_model(gp::GPModel, loc::Matrix{Float64}, data::Float64) = GP(hcat(gp.model.x,Matrix(transpose(loc))),vcat(gp.model.y, data),gp.model.mean,gp.model.kernel)

abstract type Plan end

get_plan(plan::Plan) = plan.plan
get_action(plan::Plan) = plan.action

mutable struct NullPlan <: Plan
    plan::Bool
    action::Bool
end
NullPlan() = NullPlan(false, false)

mutable struct PathPlan <: Plan
    plan::Matrix{Float64}
    action::Integer
end
get_target(p::PathPlan) = reshape(p.plan[p.action,:], 1, :)
get_plan_length(p::PathPlan) = size(p.plan)[1]
function set_target(plan::PathPlan, action::Integer)
    plan.action = action
end

mutable struct Agent
    loc::Matrix{Float64}
    speed::Float64
    model::Model
    plan::Plan
end
Human(; spawn_loc::Matrix{Float64}=[0. 0.], speed::Float64=1.0, model::Model) = Agent(spawn_loc, speed, model, NullPlan())
AUV(; spawn_loc::Matrix{Float64}=[0. 0.], speed::Float64=5.0, model::Model) = Agent(spawn_loc, speed, model, NullPlan())