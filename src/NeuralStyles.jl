module NeuralStyles

using Metalhead
using Images
using Optim
using Flux
#using CuArrays
import Flux: @epochs
#CuArrays.allowscalar(false)

include("pre.jl")
include("styles.jl")

export train

end # module
