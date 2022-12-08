<div id="top"></div>

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ThirdAILabs/Demos">
    <img src="https://www.thirdai.com/wp-content/uploads/2022/06/ThirdAI_logo.png" alt="Logo" width="150" height="">
  </a>

<h1 align="center">Demos</h1>

  <p align="center">
    Interactive notebooks for exploring ThirdAI's BOLT library.
    <br>
    <br>
    <a href="https://thirdai.com">[Website]</a>
    ·
    <a href="https://thirdai.com/docs/">[Docs]</a>
    ·
    <a href="https://github.com/ThirdAILabs/Demos/issues">[Report Bug]</a>
    ·
    <a href="https://www.thirdai.com/careers/">[We're Hiring]</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<br>
Table of Contents
<ol>
  <li>
    <a href="#👋-welcome">Welcome</a>
  </li>
  <li>
    <a href="#🚀-quickstart">Quickstart</a>
    <ul>
      <li><a href="#step-1:-downloading-a-license">Downloading a License</a></li>
      <li><a href="#step-2:-installation">Installation</a></li>
    </ul>
  </li>
  <li><a href="#🎮-usage">Usage</a></li>
  <li><a href="#📄-license">License</a></li>
  <li><a href="#🎙-contact">Contact</a></li>
</ol>

<br>



<!-- ABOUT THE PROJECT -->
# 👋 Welcome

ThirdAI's BOLT library is a deep-learning framework that leverages sparsity to enable training and deploying very large scale deep learning models on any CPU. This demo repo will help get you familiar with BOLT's [Universal Deep Transformer (UDT)](https://www.thirdai.com/universal-deep-transformers/) through interactive notebooks.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
# 🚀 Quickstart

### Step 1: Downloading a License

To use our library, you will first need to register for a free license [here](https://www.thirdai.com/try-bolt/). We will email you a unique download link for your license.

To download your license, run:

```sh
wget your_download_link ~/license.serialized
```

This will download the license file to your home directory. Please note: a valid, unexpired license must be placed in your <em>home directory</em>. If a valid license is not found, these demos will fail.

### Step 2: Installation

You can download the ThirdAI library with your package manager of choice.

For example:

```sh
pip3 install thirdai
```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
# 🎮 Usage

First, clone the demo repo:

```sh
git clone https://github.com/ThirdAILabs/Demos.git
```

Each of the demo notebooks will walk you through different aspects of the UDT API that you can run on any CPUs (ARM, AMD, Intel) and even desktops and laptops

<ul>
<li><strong>CensusIncomePrediction.ipynb</strong> shows how to build an income prediction model with ThirdAI's Universal Deep Transformer (UDT) model, our all-purpose classifier for tabular datasets.</li>
<li><strong>DistributedInferenceDatabricks.ipynb</strong> will show how to use thirdai for distributed inference on Databricks with UDT.
<li><strong>IntentClassification.ipynb</strong> will show you how to get near SOTA accuracy on most text classification via a plug and play classifier at any given budget (everything autotuned).</li>
<li><strong>FraudDetection.ipynb</strong> will show you how easy to build a fraud detection model with UDT.</li>
<li><strong>PersonalizedMovieRecommendations.ipynb</strong> will show you how to build personalization model for movie recommendation. UDT can be used to build any kind of personlization and recomnedation models with ease and deliver SOTA results.</li>
<li><strong>QueryReformulation.ipynb</strong> shows how to build a query reformulation model with UDT, providing an easy and faster (less than 1 ms) solution for query reformulation.</li>
<li><strong>SentimentAnalysis.ipynb</strong> will take you throught the process of creating a network to use during sparse training and sparse inference with the goal of predicting positive/negative sentiment.
</ul>

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
# 📄 License

Please refer to `LICENSE.txt` for more information on usage terms.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
# 🎙 Contact

ThirdAILabs - [@ThirdAILab](https://twitter.com/ThirdAILab) - [contact@thirdai.com](mailto:contact@thirdai.com)

Project Link: [https://github.com/ThirdAILabs/Demos](https://github.com/ThirdAILabs/Demos)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[forks-shield]: https://img.shields.io/github/forks/thirdailabs/demos.svg?style=for-the-badge
[forks-url]: https://github.com/ThirdAILabs/Demos/network/members
[stars-shield]: https://img.shields.io/github/stars/thirdailabs/demos.svg?style=for-the-badge
[stars-url]: https://github.com/ThirdAILabs/Demos/stargazers
[issues-shield]: https://img.shields.io/github/issues/thirdailabs/demos.svg?style=for-the-badge
[issues-url]: https://github.com/ThirdAILabs/Demos/issues
[license-shield]: https://img.shields.io/github/license/thirdailabs/demos.svg?style=for-the-badge
[license-url]: https://github.com/ThirdAILabs/Demos/blob/master/LICENSE.txt
