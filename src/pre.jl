
function preprocess_img(img, shape)
    resized_img = Images.imresize(img, (shape[2], shape[1]))
    sample = channelview(resized_img) * 256
    sample[1,:,:] -= 123.68
    sample[2,:,:] -= 116.779
    sample[3,:,:] -= 103.939
    sample = permutedims(sample, [2,3,1])
    return reshape(sample, (size(sample)[1], size(sample)[2], 3, 1))
end

function postprocess_img(img)
    img = reshape(img, (size(img)[1], size(img)[2], 3))
    img[:,:,1] += 123.68
    img[:,:,2] += 116.779
    img[:,:,3] += 103.939
	img = permutedims(img, [3,1,2])
    img = clamp.(img, 0, 255) / 255
    colorview(RGB{Float32}, img)
end
