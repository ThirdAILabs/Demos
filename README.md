<div id="top"></div>

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ThirdAILabs/Demos">
    <img src="https://www.thirdai.com/wp-content/uploads/2021/11/logo-illustrator1.png" alt="Logo" width="150" height="">
  </a>

<h1 align="center">Demos</h1>

  <p align="center">
    Interactive notebooks for exploring ThirdAI's library.
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
      <li><a href="#downloading-a-license">Downloading a License</a></li>
      <li><a href="#installation">Installation</a></li>
    </ul>
  </li>
  <li><a href="#🎮-usage">Usage</a></li>
  <li><a href="#📄-license">License</a></li>
  <li><a href="#🎙-contact">Contact</a></li>
</ol>

<br>



<!-- ABOUT THE PROJECT -->
# 👋 Welcome

ThirdAI's library is a deep-learning framework that leverages sparsity to make training and inference computationally feasible on CPUs. Our demo repo will help get you familiar with our [BOLT](https://www.thirdai.com/bolt-overview/) API through interactive notebooks.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
# 🚀 Quickstart

## Downloading a License

To use our library, you will first need to register for a free license [here](https://www.thirdai.com/try-bolt/). We will email you a unique download link for your license.

To download your license, run:

```sh
wget your_download_link ~/license.serialized
```

This will download the license file to your home directory. Please note: a valid, unexpired license must be placed in your <em>home directory</em>. If a valid license is not found, these demos will fail.

## Installation

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

Each of the demo notebooks will walk you through different aspects of the BOLT engine.

<ul>
  <li><strong>DocSearch.ipynb</strong> will show you how to use your own dataset (or one of the provided datasets) to create a simple document + query embedding model.</li>
  <li><strong>SentimentAnalysis.ipynb</strong> will take you throught the process of creating a network to use during sparse training and sparse inference with the goal of predicting positive/negative sentiment.
</ul>

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
# 📄 License

See `LICENSE.txt` for more information.

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
