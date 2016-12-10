--[[
   Principal component analysis.
   Ref: 15.2.3 PCA Algorithm Bayesian Reasoning and Machine Learning
--]]

local PCA = torch.class('unsupervised.PCA')

function PCA:__init()
end

-- data is NxD tensor.
function PCA:doPCA(data, M, inplace)
   self.inplace = inplace or false

   self.N = data:size(1)
   self.D = data:size(2)
   self.M = M or self.D

   -- Compute mean
   self.mean = data:mean(1)

   -- Mean zero the data
   local normedData = data.new()
   if self.inplace then
      normedData = data
   else
      normedData:resizeAs(data):copy(data)
   end
   local tempMean = self.mean.new()
   tempMean:resizeAs(self.mean):copy(self.mean)
   tempMean:expandAs(data)
   -- This is done for GPU efficiency you could also directly use 'tempMean'
   local expandedMean = self.mean.new()
   expandedMean:resizeAs(tempMean):copy(tempMean)
   normedData:csub(expandedMean)

   -- Compute unbiased sample covariance
   self.covar = torch.mm(normedData:t(), normedData)
   self.covar:div(self.N - 1)

   -- Eigen value decomposition for Symmetric matrix
   self.values, self.vectors = torch.symeig(self.covar, 'V') 
end
