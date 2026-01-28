#pragma once

// Static, compile-time optimized WaveNet implementation
// Based on WaveNetDynamic.h with fixed architecture parameters

#include <Eigen/Dense>
#include <Eigen/Core>
#include <array>
#include "NeuralModel.h"
#include "Activation.h"

#ifndef WAVENET_MAX_NUM_FRAMES
#define WAVENET_MAX_NUM_FRAMES 64
#endif

#ifndef LAYER_ARRAY_BUFFER_PADDING
#define LAYER_ARRAY_BUFFER_PADDING 24
#endif

namespace NeuralAudio
{
	// Static Conv1D with template parameters
	template<size_t InChannels, size_t OutChannels, size_t KernelSize, bool DoBias, size_t Dilation>
	class Conv1DStatic
	{
	private:
		std::array<Eigen::Matrix<float, OutChannels, InChannels>, KernelSize> weights;
		Eigen::Matrix<float, OutChannels, 1> bias;

	public:
		Conv1DStatic()
		{
			if constexpr (DoBias)
			{
				bias.setZero();
			}
		}

		void SetWeights(std::vector<float>::iterator& inWeights)
		{
			for (size_t i = 0; i < OutChannels; i++)
				for (size_t j = 0; j < InChannels; j++)
					for (size_t k = 0; k < KernelSize; k++)
						weights[k](i, j) = *(inWeights++);

			if constexpr (DoBias)
			{
				for (size_t i = 0; i < OutChannels; i++)
					bias(i) = *(inWeights++);
			}
		}

		inline void Process(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output, const size_t iStart, const size_t nCols) const
		{
			// Unroll the kernel loop at compile time
			[&]<size_t... Is>(std::index_sequence<Is...>)
			{
				((ProcessKernel<Is>(input, output, iStart, nCols)), ...);
			}(std::make_index_sequence<KernelSize>{});

			if constexpr (DoBias)
				output.colwise() += bias;
		}

	private:
		template<size_t K>
		inline void ProcessKernel(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output, const size_t iStart, const size_t nCols) const
		{
			constexpr size_t offset = Dilation * (K + 1 - KernelSize);
			auto& inBlock = input.middleCols(iStart + offset, nCols);

			if constexpr (K == 0)
				output.noalias() = weights[K] * inBlock;
			else
				output.noalias() += weights[K] * inBlock;
		}
	};

	// Static DenseLayer
	template<size_t InSize, size_t OutSize, bool DoBias>
	class DenseLayerStatic
	{
	private:
		Eigen::Matrix<float, OutSize, InSize> weights;
		Eigen::Matrix<float, OutSize, 1> bias;

	public:
		DenseLayerStatic()
		{
			weights.setZero();
			if constexpr (DoBias)
			{
				bias.setZero();
			}
		}

		void SetWeights(std::vector<float>::iterator& inWeights)
		{
			for (auto i = 0; i < OutSize; i++)
				for (auto j = 0; j < InSize; j++)
					weights(i, j) = *(inWeights++);

			if constexpr (DoBias)
			{
				for (auto i = 0; i < OutSize; i++)
					bias(i) = *(inWeights++);
			}
		}

		void Process(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output) const
		{
			if constexpr (DoBias)
			{
				output.noalias() = (weights * input).colwise() + bias;
			}
			else
			{
				output.noalias() = weights * input;
			}
		}

		void ProcessAcc(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output) const
		{
			if constexpr (DoBias)
			{
				output.noalias() += (weights * input).colwise() + bias;
			}
			else
			{
				output.noalias() += weights * input;
			}
		}
	};

	// Static WaveNetLayer
	template<size_t ConditionSize, size_t Channels, size_t KernelSize, size_t Dilation>
	class WaveNetLayerStatic
	{
	private:
		Conv1DStatic<Channels, Channels, KernelSize, true, Dilation> conv1D;
		DenseLayerStatic<ConditionSize, Channels, false> inputMixin;
		DenseLayerStatic<Channels, Channels, true> oneByOne;
		Eigen::Matrix<float, Channels, WAVENET_MAX_NUM_FRAMES> state;
		Eigen::MatrixXf layerBuffer;

	public:
		static constexpr size_t ReceptiveFieldSize = (KernelSize - 1) * Dilation;
		size_t bufferStart;

		WaveNetLayerStatic()
		{
			state.setZero();
		}

		Eigen::MatrixXf& GetLayerBuffer()
		{
			return layerBuffer;
		}

		void AllocBuffer(size_t allocNum)
		{
			constexpr size_t size = ReceptiveFieldSize + ((LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES);

			layerBuffer.resize(Channels, size);
			layerBuffer.setZero();

#if (LAYER_ARRAY_BUFFER_PADDING == 0)
			bufferStart = ReceptiveFieldSize;
#else
			bufferStart = size - (WAVENET_MAX_NUM_FRAMES * ((allocNum % LAYER_ARRAY_BUFFER_PADDING) + 1));
#endif
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			conv1D.SetWeights(weights);
			inputMixin.SetWeights(weights);
			oneByOne.SetWeights(weights);
		}

		void SetMaxFrames(const size_t maxFrames)
		{
			(void)maxFrames;
		}

		void AdvanceFrames(const size_t numFrames)
		{
			bufferStart += numFrames;

			if ((int)(bufferStart + WAVENET_MAX_NUM_FRAMES) > layerBuffer.cols())
				RewindBuffer();
		}

		void RewindBuffer()
		{
			constexpr size_t start = ReceptiveFieldSize;

			layerBuffer.middleCols(start - ReceptiveFieldSize, ReceptiveFieldSize) = layerBuffer.middleCols(bufferStart - ReceptiveFieldSize, ReceptiveFieldSize);

			bufferStart = start;
		}

		void CopyBuffer()
		{
			for (size_t offset = 1; offset < ReceptiveFieldSize + 1; offset++)
			{
				layerBuffer.col(bufferStart - offset) = layerBuffer.col(bufferStart);
			}
		}

		void Process(const Eigen::Ref<const Eigen::MatrixXf>& condition, Eigen::Ref<Eigen::MatrixXf> headInput, Eigen::Ref<Eigen::MatrixXf> output, const size_t outputStart, const size_t numFrames)
		{
			auto block = state.leftCols(numFrames);

			conv1D.Process(layerBuffer, block, bufferStart, numFrames);

			inputMixin.ProcessAcc(condition, block);

			// Apply tanh activation
			float* data = block.data();
			const auto size = block.rows() * block.cols();

			for (auto pos = 0; pos < size; pos++)
			{
				data[pos] = WAVENET_MATH::Tanh(data[pos]);
			}

			headInput.noalias() += block.template topRows<Channels>();

			oneByOne.Process(block.template topRows<Channels>(), output.middleCols(outputStart, numFrames));

			output.middleCols(outputStart, numFrames).noalias() += layerBuffer.middleCols(bufferStart, numFrames);

			AdvanceFrames(numFrames);
		}
	};

	// Static WaveNetLayerArray with compile-time known dilations
	template<size_t InputSize, size_t ConditionSize, size_t HeadSize, size_t Channels, size_t KernelSize, bool HasHeadBias, size_t... Dilations>
	class WaveNetLayerArrayStatic
	{
	private:
		static constexpr size_t NumLayers = sizeof...(Dilations);
		
		std::tuple<WaveNetLayerStatic<ConditionSize, Channels, KernelSize, Dilations>...> layers;
		DenseLayerStatic<InputSize, Channels, false> rechannel;
		DenseLayerStatic<Channels, HeadSize, HasHeadBias> headRechannel;
		Eigen::MatrixXf arrayOutputs;
		Eigen::MatrixXf headOutputs;

	public:
		WaveNetLayerArrayStatic()
		{
			arrayOutputs.resize(Channels, WAVENET_MAX_NUM_FRAMES);
			headOutputs.resize(HeadSize, WAVENET_MAX_NUM_FRAMES);
			arrayOutputs.setZero();
			headOutputs.setZero();
		}

		Eigen::MatrixXf& GetArrayOutputs()
		{
			return arrayOutputs;
		}

		Eigen::MatrixXf& GetHeadOutputs()
		{
			return headOutputs;
		}

		constexpr size_t GetNumChannels() const
		{
			return Channels;
		}

		size_t AllocBuffers(size_t allocNum)
		{
			std::apply([&allocNum](auto&... layer) {
				((layer.AllocBuffer(allocNum++)), ...);
			}, layers);

			return allocNum;
		}

		void SetMaxFrames(const size_t maxFrames)
		{
			std::apply([maxFrames](auto&... layer) {
				((layer.SetMaxFrames(maxFrames)), ...);
			}, layers);
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			rechannel.SetWeights(weights);

			std::apply([&weights](auto&... layer) {
				((layer.SetWeights(weights)), ...);
			}, layers);

			headRechannel.SetWeights(weights);
		}

		void Prewarm(const Eigen::MatrixXf& layerInputs, const Eigen::MatrixXf& condition, Eigen::Ref<Eigen::MatrixXf> const& headInputs)
		{
			rechannel.Process(layerInputs, std::get<0>(layers).GetLayerBuffer().middleCols(std::get<0>(layers).bufferStart, 1));

			PrewarmLayers<0>(condition, headInputs);

			headRechannel.Process(headInputs, headOutputs.leftCols(1));
		}

		void Process(const Eigen::MatrixXf& layerInputs, const Eigen::MatrixXf& condition, Eigen::Ref<Eigen::MatrixXf> headInputs, const size_t numFrames)
		{
			rechannel.Process(layerInputs, std::get<0>(layers).GetLayerBuffer().middleCols(std::get<0>(layers).bufferStart, numFrames));

			ProcessLayers<0>(condition, headInputs, numFrames);

			headRechannel.Process(headInputs, headOutputs.leftCols(numFrames));
		}

	private:
		template<size_t LayerIndex>
		void PrewarmLayers(const Eigen::MatrixXf& condition, Eigen::Ref<Eigen::MatrixXf> const& headInputs)
		{
			auto& layer = std::get<LayerIndex>(layers);
			layer.CopyBuffer();

			if constexpr (LayerIndex == NumLayers - 1)
			{
				layer.Process(condition, headInputs, arrayOutputs, 0, 1);
			}
			else
			{
				auto& nextLayer = std::get<LayerIndex + 1>(layers);
				layer.Process(condition, headInputs, nextLayer.GetLayerBuffer(), nextLayer.bufferStart, 1);
				
				PrewarmLayers<LayerIndex + 1>(condition, headInputs);
			}
		}

		template<size_t LayerIndex>
		void ProcessLayers(const Eigen::MatrixXf& condition, Eigen::Ref<Eigen::MatrixXf> headInputs, const size_t numFrames)
		{
			auto& layer = std::get<LayerIndex>(layers);

			if constexpr (LayerIndex == NumLayers - 1)
			{
				layer.Process(condition, headInputs, arrayOutputs, 0, numFrames);
			}
			else
			{
				auto& nextLayer = std::get<LayerIndex + 1>(layers);
				layer.Process(condition, headInputs, nextLayer.GetLayerBuffer(), nextLayer.bufferStart, numFrames);
				
				ProcessLayers<LayerIndex + 1>(condition, headInputs, numFrames);
			}
		}
	};

	// Type aliases for the two model architectures
	// Large model: 21 layers, 8 channels, dilations [1,1,3,11,29,83,233] x3
	using WaveNetLayerArrayLarge = WaveNetLayerArrayStatic<
		1, 1, 8, 8, 6, true,
		1, 1, 3, 11, 29, 83, 233,
		1, 1, 3, 11, 29, 83, 233,
		1, 1, 3, 11, 29, 83, 233
	>;

	// Small model: 18 layers, 4 channels, dilations [1,2,7,23,73,251] x3
	using WaveNetLayerArraySmall = WaveNetLayerArrayStatic<
		1, 1, 4, 4, 6, true,
		1, 2, 7, 23, 73, 251,
		1, 2, 7, 23, 73, 251,
		1, 2, 7, 23, 73, 251
	>;

	// Static WaveNetModel template
	template<typename LayerArrayType, size_t Channels>
	class WaveNetModelStatic : public NeuralModel
	{
	private:
		LayerArrayType layerArray;
		Eigen::MatrixXf headArray;
		float headScale;
		size_t maxFrames;

	public:
		WaveNetModelStatic()
		{
			layerArray.AllocBuffers(0);
			headArray.resize(Channels, WAVENET_MAX_NUM_FRAMES);
			headArray.setZero();
			headScale = 1.0f;
		}

		void SetWeights(std::vector<float> weights)
		{
			std::vector<float>::iterator it = weights.begin();

			layerArray.SetWeights(it);

			headScale = *(it++);

			assert(std::distance(weights.begin(), it) == (long)weights.size());
		}

		size_t GetMaxFrames() const
		{
			return maxFrames;
		}

		void SetMaxFrames(const size_t frames)
		{
			this->maxFrames = frames;

			if (this->maxFrames > WAVENET_MAX_NUM_FRAMES)
				this->maxFrames = WAVENET_MAX_NUM_FRAMES;

			layerArray.SetMaxFrames(this->maxFrames);
		}

		void Prewarm() override
		{
			float input = 0;

			auto condition = Eigen::Map<const Eigen::Matrix<float, 1, -1>>(&input, 1, 1);

			layerArray.Prewarm(condition, condition, headArray.leftCols(1));
		}

		void Process(float* input, float* output, size_t numFrames) override
		{
			auto condition = Eigen::Map<const Eigen::MatrixXf>(input, 1, numFrames);

			headArray.setZero();

			layerArray.Process(condition, condition, headArray.leftCols(numFrames), numFrames);

			const auto& finalHeadArray = layerArray.GetHeadOutputs();

			auto out = Eigen::Map<Eigen::Matrix<float, 1, -1>>(output, 1, numFrames);

			out.noalias() = headScale * finalHeadArray.leftCols(numFrames);
		}
		
		bool IsStatic() override
		{
			return true;
		}
	};

	// Concrete classes for registration - these are complete types
	class WaveNetModelLarge : public WaveNetModelStatic<WaveNetLayerArrayLarge, 8>
	{
	public:
		WaveNetModelLarge() = default;
	};

	class WaveNetModelSmall : public WaveNetModelStatic<WaveNetLayerArraySmall, 4>
	{
	public:
		WaveNetModelSmall() = default;
	};
}
