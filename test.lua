require 'torch'

local unitTests = torch.TestSuite()
local precision = 1e-5
local mytester

-- Unit test: Principal Component Analysis
function unitTests.PCA()
   print("Testing PCA")
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}

   local N = 100
   local D = 5
   local M = 2
   local data = torch.rand(N, D)

   local pca = unsupervised.PCA(M)
   pca:train(data)

   -- Mean must match
   local mean = data:mean(1)
   mytester:assertTensorEq(mean, pca.mean, precision,
                           "Incorrect Mean vectors.")

   -- Eigen vectors should be unit vectors
   local norms = pca.vectors:norm(2, 2)
   mytester:assertTensorEq(norms, torch.ones(D, 1), precision,
                           "Eigen vectors don't have unit length.")
  
   local Y = pca:project(data)
   local yMean = Y:mean(1)
   mytester:assertTensorEq(yMean, torch.zeros(1, M), precision,
         "Projected vectors Mean should be very close to zero.")
end

function unsupervised.test(tests)
   mytester = torch.Tester()
   mytester:add(unitTests)
   math.randomseed(os.time())
   mytester:run(tests)
end
