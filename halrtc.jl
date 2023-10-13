using Plots
using Images
using FileIO
using LinearAlgebra
using SchattenNorms
using Statistics
using DelimitedFiles
ci = CartesianIndex

function mat2ten(A, n, dims)
    m = setdiff(1:length(dims), n)
    X = reshape(A, [dims[n]; dims[m]]...)
    permutedims(X, invperm([n; m]))
end

function ten2mat(X, n)
    sz = size(X)
    m = setdiff(1:3, n)
    reshape(permutedims(X, [n;m]), sz[n], prod(sz[m]))
end

function HaLRTC(T::Array{Float64, 3},
                Omega::Vector, 
                rev_Omega::Vector,
                alpha = [1, 1, 1e-3], 
                rho = 1e-6, 
                maxiter = 500, 
                epsilon = 8e-7)
    X = deepcopy(T)
    X[rev_Omega] .= mean(T[Omega])
    dim = [size(T)[1], size(T)[2], size(T)[3]]

    if dim[1] == 256 epsilon /= 4.0 end

    Y = [X, X, X]
    M = [zeros(dim[1], dim[2], dim[3]),
         zeros(dim[1], dim[2], dim[3]),
         zeros(dim[1], dim[2], dim[3])]
    Msum = zeros(dim[1], dim[2], dim[3])
    Ysum = zeros(dim[1], dim[2], dim[3])
    errs = zeros(maxiter)
    normT = norm(T)
    println("HaLRTC begins.")
    for k=1:maxiter
        if k%20==0
            print("Iteration: ", k, '\t')
            println("Error: ", errs[k-1])
        end
        rho *= 1.05
        Msum *= 0.0
        Ysum *= 0.0
        for i=1:3
            matrix = ten2mat(X-M[i]/rho, i)
            U, S, V = svd(matrix)
            for j in eachindex(S)
                S[j] = max(S[j]-alpha[i]/rho, 0)
            end
            Y[i] = mat2ten(U*Diagonal(S)*V', i, dim)
            Msum += M[i]
            Ysum += Y[i]
        end
        lastX = X
        X = (Msum+rho*Ysum)/(3*rho)
        X[Omega] = T[Omega]
        for i=1:3
            M[i] += rho*(Y[i]-X)
        end
        errs[k] = norm(X-lastX)/normT
        if errs[k] < epsilon
            break
        end
    end
    return X, errs
end

function run(imgname, miss_percent, dims)
    println("Running HaLRTC for \"", imgname, "\" with ", miss_percent, "% missing voxels.")

    ori = load("./originals/"*imgname*".png")
    raw = load("./test_imgs/"*imgname*miss_percent*".png")

    input = zeros(dims[1], dims[2], 3)
    origin = zeros(dims[1], dims[2], 3)

    mask = ci{3}[]
    rev_mask = ci{3}[]
    for i=1:dims[1]
        for j=1:dims[2]
            c = raw[i, j]
            input[i, j, 1] = c.r*255
            input[i, j, 2] = c.g*255
            input[i, j, 3] = c.b*255
            origin[i, j, 1] = ori[i, j].r*255
            origin[i, j, 2] = ori[i, j].g*255
            origin[i, j, 3] = ori[i, j].b*255
            for k=1:3
                if input[i, j, k] < 255
                    push!(mask, ci(i, j, k))
                else
                    push!(rev_mask, ci(i, j, k))
                end
            end
        end
    end

    ans, errs = HaLRTC(origin, mask, rev_mask)

    img = zeros(RGB{Float64}, dims[1], dims[2])
    for i=1:dims[1]
        for j=1:dims[2]
            r = trunc(Int, ans[i, j, 1])
            g = trunc(Int, ans[i, j, 2])
            b = trunc(Int, ans[i, j, 3])
            if r > 255 r = 255 end
            if r < 0 r = 0 end
            if g > 255 g = 255 end
            if g < 0 g = 0 end
            if b > 255 b = 255 end
            if b < 0 b = 0 end
            img[i, j] = RGB(r/255.0, g/255.0, b/255.0)
        end
    end
    save("results/"*imgname*miss_percent*"ans.png", img)
    savefig(plot(errs, legend = false), "results/"*imgname*miss_percent*"errs.png")
end

for imgname in ["airplane", 
                "barbara", 
                "couple", 
                "facade", 
                "house", 
                "jellybeans", 
                "lenna", 
                "mandrill", 
                "peppers", 
                "sailboat", 
                "splash", 
                "tree"]
    for miss_percent in ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
        if imgname in ["airplane", 
                       "lenna",
                       "mandrill", 
                       "peppers", 
                       "sailboat", 
                       "splash"]
            run(imgname, miss_percent, [512 512])
        else
            run(imgname, miss_percent, [256 256])
        end
    end
end
