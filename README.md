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

### Step 1: Running a UDT Demo

All of our UDT capability demos are mirrored to Google Colab, so you can immediately run any of them by clicking the associated link:

<ul>
<li><strong>CensusIncomePrediction.ipynb</strong> shows how to build an income prediction model with ThirdAI's Universal Deep Transformer (UDT) model, our all-purpose classifier for tabular datasets.
<br>https://colab.research.google.com/github/ThirdAILabs/Demos/blob/main/classification/CensusIncomePrediction.ipynb
</li>
<li><strong>ClickThroughPrediction.ipynb</strong> shows how you can use UDT to achieve SOTA AUC on Click Through Prediction.
<br>https://githubtocolab.com/ThirdAILabs/Demos/blob/main/classification/ClickThroughPrediction.ipynb
</li>
<li><strong>EmbeddingsAndColdStart.ipynb</strong> takes care of your most NLP, search, and recommendations needs on unstructured raw text.  Learn with simple commands how to train large neural models on raw text to perform search, recommendations, and generate entity emebeddings as well as embeddings for any text. Yes, all (training, inference, and retraining) on simple CPUs.   
<br>https://githubtocolab.com/ThirdAILabs/Demos/blob/main/embeddings/EmbeddingsAndColdStart.ipynb
</li>
<li><strong>IntentClassification.ipynb</strong> will show you how to get near SOTA accuracy on most text classification via a plug and play classifier at any given budget (everything autotuned).
<br>https://githubtocolab.com/ThirdAILabs/Demos/blob/main/classification/IntentClassification.ipynb
</li>
<li><strong>FraudDetection.ipynb</strong> will show you how easy to build a fraud detection model with UDT.
<br>https://githubtocolab.com/ThirdAILabs/Demos/blob/main/classification/FraudDetection.ipynb
</li>
<li><strong>PersonalizedMovieRecommendations.ipynb</strong> will show you how to build personalization model for movie recommendation. UDT can be used to build any kind of personlization and recomnedation models with ease and deliver SOTA results.
<br>https://githubtocolab.com/ThirdAILabs/Demos/blob/main/personlization_and_recommendation/PersonalizedMovieRecommendations.ipynb
</li>
<li><strong>QueryReformulation.ipynb</strong> shows how to build a query reformulation model with UDT, providing an easy and faster (less than 1 ms) solution for query reformulation.
<br>https://colab.research.google.com/github/ThirdAILabs/Demos/blob/main/QueryReformulation.ipynb
</li>
<li><strong>SentimentAnalysis.ipynb</strong> will take you through the process of creating a network to use during sparse training and sparse inference with the goal of predicting positive/negative sentiment.
<br>https://githubtocolab.com/ThirdAILabs/Demos/blob/main/classification/SentimentAnalysis.ipynb
</li>

<li><strong>TrainingDistributedUDT.ipynb</strong> shows how you can use ThirdAI's UDT in distributed setting using Ray cluster. For this demo, we are using clinc-small for training and evaluation.
<br>https://github.com/ThirdAILabs/Demos/blob/main/distributed/TrainingDistributedUDT.ipynb
</li>
</ul>

You can also clone this repo and run any of these demo notebooks on any CPU (ARM, AMD, Intel), and even desktops and laptops

### Step 2: Integrating UDT with an Existing Platform

We also have demos explaining how to integrate UDT with different platforms you may already be comfortable with:

<ul>
<li><strong>integrations/DeployThirdaiwithDatabricks.ipynb</strong> will show how to use thirdai for inference on Databricks with UDT.</li>
</ul>

<p align="right">(<a href="#top">back to top</a>)</p>

### Step 3: Trying UDT on Your Own Dataset

Each of these notebooks has an API key that will only work on the dataset in the demo. If you want to try out ThirdAI on your own dataset, simply register for a free license [here](https://www.thirdai.com/try-bolt/). We look forward to hearing from you!

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
