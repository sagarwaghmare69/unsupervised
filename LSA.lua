--[[
   Latent Semantic Analysis. Primaryly used in Document Analysis.
   LSA is similar to PCA-SVD. It used SVD for computing the eigen
   values and vectors. Another difference is that the projections
   are scaled such that it has unit covariance.

   Ref: 15.4 Bayesian Reasoning and Machine Learning
--]]

local LSA, parent = torch.class("unsupervised.LSA",
                                "unsupervised.ParentModule")

-- If rescale is false then this is same as PCA-SVD
function LSA:__init(M, rescale)
   parent.__init(self)
   self.M = M
   if rescale == nil then
      self.rescale = true -- Default rescale projections
   else
      self.rescale = rescale
   end
end

function LSA:train(X, inplace)
   self.N = X:size(1)
   self.D = X:size(2)
   local inplace = inplace or false

   -- Compute mean
   self.mean = X:mean(1)

   -- Mean zero the X
   local normedX = self:_meanZero(X, self.mean, inplace)

   -- SVD
   local U, S, V = torch.svd(normedX:t())

   -- Eigen values and eigen vectors
   self.Dvalues = S
   self.vectors = U
   -- Eigen Values = square of singular values
   self.values = self.Dvalues:clone()
   self.values:pow(2)

   --[[ SVD output is in descending order of eigen values hence
        no need to sort eigen values and eigen vectors --]]
   self.trained = true
end

-- Project to M dimensions
function LSA:project(X, meanZeroed, inplace, M)
   assert(self.trained, "LSA not trained.")
   local meanZeroed = meanZeroed or false
   local inplace = inplace or false
   local M = self.M or M

   assert(M<=self.D, "M should be less than or equal to self.D.")

   -- Mean zero X
   local normedX = X.new()
   if meanZeroed then
      normedX = X
   else
      normedX = self:_meanZero(X, self.mean, inplace)
   end

   local U = self.vectors[{{1, M}}]:t()
   local Dvalues = self.Dvalues[{{1, M}}]
   local DinvValues = Dvalues:clone():fill(1):cdiv(Dvalues)
   local Dinv = torch.diag(DinvValues)

   local tempY = torch.mm(normedX, U)
   local Y = tempY.new()
   if self.rescale then
      Y = torch.mm(tempY, Dinv)
      Y:mul(math.sqrt(self.N-1))
   else
      Y = tempY
   end
   return Y
end

-- Reconstruct X using U and D
function LSA:reconstruct(X, meanZeroed, inplace, M)
   local reconX = X.new()
   local M = self.M or M
   local Y = self:project(X, M, meanZeroed, inplace)

   -- Pick first M eigen vectors
   local U = self.vectors[{{1, M}}]

   if self.rescale then
      local D = torch.diag(self.Dvalues[{{1, M}}])
      local tempReconX = torch.mm(Y, D)
      reconX = torch.mm(tempReconX, U)
      reconX:div(math.sqrt(self.N-1))
   else
      -- Similar to PCA
      reconX = torch.mm(Y, U) 
   end

   -- Add mean
   local tempMean = self.mean.new()
   tempMean:resizeAs(self.mean):copy(self.mean)
   tempMean:expandAs(tempMean, reconX)
   local expandedMean = self.mean.new()
   expandedMean:resizeAs(tempMean):copy(tempMean)
   reconX:add(expandedMean)
   return reconX
end
