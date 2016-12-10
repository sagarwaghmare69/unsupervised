--[[
   Principal component analysis.
   Ref: 15.2.3 PCA Algorithm Bayesian Reasoning and Machine Learning
--]]

local PCA, parent = torch.class('unsupervised.PCA',
                                'unsupervised.ParentModule')

function PCA:__init()
   self.train = false
end

-- data is NxD tensor.
function PCA:process(data, M, inplace)
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
   tempMean:expandAs(tempMean, data)
   -- This is done for GPU efficiency you could also directly use 'tempMean'
   local expandedMean = self.mean.new()
   expandedMean:resizeAs(tempMean):copy(tempMean)
   normedData:csub(expandedMean)

   -- Compute unbiased sample covariance
   self.covar = torch.mm(normedData:t(), normedData)
   self.covar:div(self.N - 1)

   -- Eigen value decomposition for Symmetric matrix
   local values, vectors = torch.symeig(self.covar, 'V')

   --[[ Output returned by symeig are in ascending order or eigen values
        hence reversing the order --]]
   local indices = torch.linspace(self.D, 1, self.D):long()
   self.values = values.new()
   self.vectors = vectors.new()
   self.values:index(values, 1, indices)
   self.vectors:index(vectors, 1, indices)
   self.train = true
end
