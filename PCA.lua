--[[
   Principal component analysis.
   Ref: 15.2.3 PCA Algorithm Bayesian Reasoning and Machine Learning
--]]

local PCA, parent = torch.class('unsupervised.PCA',
                                'unsupervised.ParentModule')

function PCA:__init()
   self.trained = false
end

function PCA:_meanZero(X, mean, inplace)
   local inplace = inplace or false

   -- Mean zero the X
   local normedX = X.new()
   if inplace then
      normedX = X
   else
      normedX:resizeAs(X):copy(X)
   end
   local tempMean = mean.new()
   tempMean:resizeAs(mean):copy(mean)
   tempMean:expandAs(tempMean, normedX)
   -- This is done for GPU efficiency you could also directly use 'tempMean'
   local expandedMean = mean.new()
   expandedMean:resizeAs(tempMean):copy(tempMean)
   normedX:csub(expandedMean)
   return normedX
end

-- data is NxD tensor.
function PCA:train(X, M, inplace)
   self.N = X:size(1)
   self.D = X:size(2)
   self.M = M or self.D
   local inplace = inplace or false

   -- Compute mean
   self.mean = X:mean(1)

   -- Mean zero the X
   local normedX = self:_meanZero(X, self.mean, inplace)

   -- Compute unbiased sample covariance
   self.covar = torch.mm(normedX:t(), normedX)
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
   self.trained = true
end

-- Project to PCA dimensions
function PCA:project(input, meanZeroed, inplace)

   local meanZeroed = meanZeroed or false
   local inplace = inplace or false

   if inplace then
      normedX = X
   else
      normedX:resizeAs(X):copy(X)
   end

end
