module Styles 

include("pre.jl")

using Metalhead
using Images
using Optim
using Flux
using CuArrays
import Flux: @epochs
CuArrays.allowscalar(false)

export train

"""
Main training driver function
"""
function train(con_path, sty_path, n_epoch = 140)

    info("Loading data ....")
    content = load(con_path) 
    orig_size = size(content)
    content = preprocess_img(content, (224, 224)) |> gpu

    style = load(sty_path) 
    style = preprocess_img(style, (224, 224)) |> gpu
    info("Loaded data")

    # Random starting point 
    # generated = rand(eltype(content), size(content)) * 256
    generated = 0.4f0*content + gpu(0.6f0*rand(-20.0f0:0.01f0:20.0f0, size(content)))
    generated = generated |> gpu
    generated = generated |> param
    
    m = VGG19() |> gpu

    STYLE_LAYERS = 
    [(1,  0.2),
     (4,  0.2),
     (7,  0.2),
     (12, 0.2),
     (17, 0.2)]


    #optimize(x -> compute_cost(m, content, style, x, STYLE_LAYERS), 
    #         generated, GradientDescent(), Optim.Options(show_trace = true, iterations = 2))
    loss() = compute_cost(m, content, style, generated, STYLE_LAYERS)

    @epochs n_epoch Flux.train!(loss, 
                                Iterators.repeated((), 1), ADAM([generated], 2))

    output = generated.data |> collect |> postprocess_img
    output = imresize(output, orig_size)

    save("output.jpg", output)

    output
end

function compute_cost(m, content, style, generated, STYLE_LAYERS)
    act_content = forward_pass(m, content)
    act_style = forward_pass(m, style)
    act_generated = forward_pass(m, generated)
    
    J_content = compute_content_cost(act_content, act_generated)
    println("Content cost = $J_content")
    J_style = compute_style_cost(act_content, act_generated, STYLE_LAYERS)
    println("Style cost = $J_style")
    J = compute_total_cost(J_content, J_style) 
    println("Total cost = $J")
    println("---------------")
    J
end

function compute_content_cost(act_content, act_generated)
    _content = act_content[13]
    _generated = act_generated[13]
    cost = 0.
    #=for i = 1:endof(_content)
        cost += (_content[i] - _generated[i]) ^ 2
    end=#
    cost = sum(abs2, _content - _generated)
    cost /= (prod(size(_content)) * 4)
end

function compute_style_cost(act_style, act_generated, style_layers) 
    J = 0.
    for (layer, coeff) in style_layers
        _style = act_style[layer]
        _generated = act_generated[layer]
        cost = compute_layer_style_cost(_style, _generated)
        J += coeff*cost
    end
    J
end


function compute_layer_style_cost(style, generated)

    m, n, c, _ = size(style)
    style_unrolled = reshape(style, m * n, c)
    generated_unrolled = reshape(generated, m * n, c)

    GS = gram_matrix(style_unrolled)
    GG = gram_matrix(generated_unrolled)

    cost = 0.
    #=for i = 1:endof(GS)
        cost += (GS[i] - GG[i]) ^ 2
    end=#
    cost = sum(abs2, GS - GG)

    cost /= (4 * c^2 * (m*n)^2)
    # sum(abs2, GS - GG) / (4 * c * (m*n)^2)
end

function compute_total_cost(content_cost, style_cost, α = 10, β = 40)
    α*content_cost + β*style_cost
end

function gram_matrix(A)
    #A * transpose(A)
    #A_mul_Bt(A, A)
    A'A
end

function forward_pass(m, arr)
    
    n = m.layers.layers |> length
    act = []
    
    input = arr
    for i = 1:n 
        output = m.layers[i](input)
        push!(act, output)
        input = output
    end

    act
end

end
