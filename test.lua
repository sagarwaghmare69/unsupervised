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

   local pca = unsupervised.PCA()
   pca:train(data)

   local mean = data:mean(1)

   mytester:assertTensorEq(mean, pca.mean, precision,
                           "Incorrect Mean vectors.")
end

function unsupervised.test(tests)
   mytester = torch.Tester()
   mytester:add(unitTests)
   math.randomseed(os.time())
   mytester:run(tests)
end
