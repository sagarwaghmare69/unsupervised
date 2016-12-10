--[[
   Principal component analysis.
   Ref: 15.2.3 PCA Algorithm Bayesian Reasoning and Machine Learning
--]]

local PCA, parent = torch.class('unsupervised.PCA',
                                'unsupervised.ParentModule')

function PCA:__init(M)
   self.trained = false
   self.M = M
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
function PCA:train(X, inplace)
   self.N = X:size(1)
   self.D = X:size(2)
   local inplace = inplace or false

   -- Compute mean
   self.mean = X:mean(1)

   -- Mean zero the X
   local normedX = self:_meanZero(X, self.mean, inplace)

   -- Compute unbiased sample covariance
   self.covar = torch.mm(normedX:t(), normedX)
   self.covar:div(self.N - 1)

   -- Eigen value decomposition for Symmetric matrix
   -- Eigenvectors are transposed here hence each row is a eigen vector
   local values, vectors = torch.symeig(self.covar, 'V')

   --[[ Output returned by symeig are in ascending order or eigen values
        hence reversing the order --]]
   local indices
   self.values, indices = values:sort(true)
   self.vectors = vectors.new()
   self.vectors:index(vectors, 1, indices)
   self.trained = true
end

-- Project to PCA dimensions
function PCA:project(X, M, meanZeroed, inplace)
   assert(self.trained, "PCA not trained.")
   local meanZeroed = meanZeroed or false
   local inplace = inplace or false
   M = self.M or M

   assert(M<=self.D, "M should be less than or equal to self.D.")

   -- Mean zero X
   local normedX = X.new()
   if meanZeroed then
      normedX = X
   else
      normedX = self:_meanZero(X, self.mean, inplace)
   end

   -- Pick first M eigen vectors
   local V = self.vectors[{{1, M}}]:t()

   -- Original Equation is E_t(M, D) * normedX(D,N) = (M, N)
   -- We do normedX(N,D) * V_t(D,M) = (N,M)
   -- V = E_t
   local Y = torch.mm(normedX, V)
   return Y
end
