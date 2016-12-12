-- th -lunsupervised -e "unsupervised.test{'function_name_in_test.lua'}"

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

   local reconX = pca:reconstruct(data)
   mytester:assertGeneralEq(data:size(), reconX:size(), 0,
                           "X and reconX should have same size.")
end

-- Unit test: Latent Semantic Analysis
function unitTests.LSA()
   print("Testing PCA")
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}

   local N = 1000
   local D = 5
   local M = 2
   local data = torch.rand(N, D)

   local lsa = unsupervised.LSA(M)
   lsa:train(data)

   -- Mean must match
   local mean = data:mean(1)
   mytester:assertTensorEq(mean, lsa.mean, precision,
                           "Incorrect Mean vectors.")

   -- Eigen vectors should be unit vectors
   local norms = lsa.vectors:norm(2, 2)
   mytester:assertTensorEq(norms, torch.ones(D, 1), precision,
                           "Eigen vectors don't have unit length.")
  
   -- Mean of projected samples should be close to zero  
   local Y = lsa:project(data)
   local yMean = Y:mean(1)
   mytester:assertTensorEq(yMean, torch.zeros(1, M), precision,
         "Projected vectors Mean should be very close to zero.")

   -- Std of projected samples should be close to one
   local yStd = Y:std(1)
   mytester:assertTensorEq(yStd, torch.ones(1, M), 0.1,
         "Projected vectors Std should be very close to one.")

   local reconX = lsa:reconstruct(data)
   mytester:assertGeneralEq(data:size(), reconX:size(), 0,
                           "X and reconX should have same size.")

   -- Rescaling disabled
   local lsa = unsupervised.LSA(M, false)
   lsa:train(data)

    -- Mean must match
   local mean = data:mean(1)
   mytester:assertTensorEq(mean, lsa.mean, precision,
                           "Incorrect Mean vectors.")

   -- Eigen vectors should be unit vectors
   local norms = lsa.vectors:norm(2, 2)
   mytester:assertTensorEq(norms, torch.ones(D, 1), precision,
                           "Eigen vectors don't have unit length.")

   -- Mean of projected samples should be close to zero  
   local Y = lsa:project(data)
   local yMean = Y:mean(1)
   mytester:assertTensorEq(yMean, torch.zeros(1, M), precision,
         "Projected vectors Mean should be very close to zero.")

   -- Std should not be close to 1 since rescaling is disabled
   local yStd = Y:std(1)
   mytester:assertTensorNe(yStd, torch.zeros(1, M), 0.2,
         "Projected vectors Std should not be close to one.")

   local reconX = lsa:reconstruct(data)
   mytester:assertGeneralEq(data:size(), reconX:size(), 0,
                           "X and reconX should have same size.")
end


function unsupervised.test(tests)
   mytester = torch.Tester()
   mytester:add(unitTests)
   math.randomseed(os.time())
   mytester:run(tests)
end
