base) vivek@pop-os:\~/Desktop/rover/ERC-3-earthrover-challenge$ tree  
.  
├── baseline.py  
├── chat\_handoff.md  
├── chat\_session\_reconstructed.md  
├── CONTEXT.md  
├── data  
│   ├── corrider\_db  
│   │   ├── config.json  
│   │   ├── descriptors.npz  
│   │   ├── navigation\_graph.json  
│   │   └── place\_graph.json  
│   ├── corrider\_db\_step5  
│   │   ├── config.json  
│   │   ├── descriptors.npz  
│   │   ├── navigation\_graph.json  
│   │   ├── place\_graph.json  
│   │   └── temporal\_eval.json  
│   ├── corrider\_extracted  
│   │   ├── front\_images  
│   │   │   ├── 000000.png  
│   │   │   ├── 000001.png  
│   │   │   ├── 000002.png  
│   │   │   ├── 000003.png  
│   │   │   ├── 000004.png  
│   │   │   ├── 000005.png  
│   │   │   ├── 000006.png  
│   │   │   ├── 000007.png  
│   │   │   ├── 000008.png  
│   │   │   ├── 000009.png  
│   │   │   ├── 000010.png  
│   │   │   ├── 000011.png  
│   │   │   ├── 000012.png  
│   │   │   ├── 000013.png  
│   │   │   ├── 000014.png  
│   │   │   ├── 000015.png  
│   │

│   │   └── metadata  
│   │       ├── accels.csv  
│   │       ├── controls.csv  
│   │       ├── data\_info.json  
│   │       ├── front\_frames.csv  
│   │       ├── gyros.csv  
│   │       ├── mags.csv  
│   │       ├── rpms.csv  
│   │       ├── summary.json  
│   │       └── telemetry.csv  
│   └── corrider.h5  
├── docs  
│   ├── CONTEXT.md  
│   ├── discoveries.tex  
│   ├── \[ICRA 2026 Competition\] The Earth Rover Challenge 3.pdf  
│   ├── indoor\_navigation\_strategy.tex  
│   ├── known\_corridor\_runtime\_plan.tex  
│   ├── laserfocus.tex  
│   ├── march19.tex  
│   ├── MBRA\_ALGORITHM\_SPEC.md  
│   ├── MBRA\_CODE\_FILE\_BY\_FILE.md  
│   ├── nyu\_indoor\_track.tex  
│   ├── our\_approach.tex  
│   ├── stack\_breakdown.tex  
│   └── team\_start\_here.tex  
├── earth-rovers-sdk  
│   ├── assets  
│   │   ├── axis.jpg  
│   │   └── v5.2.png  
│   ├── browser\_service.py  
│   ├── docker-compose.yml  
│   ├── Dockerfile  
│   ├── examples  
│   │   ├── front\_example.py  
│   │   ├── front\_test.html  
│   │   ├── intervention\_manager.html  
│   │   ├── \_\_pycache\_\_  
│   │   │   ├── simple\_control.cpython-313.pyc  
│   │   │   └── teleop\_legacy.cpython-313.pyc  
│   │   ├── rear\_example.py  
│   │   ├── rear\_test.html  
│   │   ├── simple\_control.py  
│   │   ├── teleop\_legacy.py  
│   │   ├── test\_with\_controls\_mini.html  
│   │   └── test\_with\_controls\_zero.html  
│   ├── index.html  
│   ├── main.py  
│   ├── README.md  
│   ├── requirements.txt  
│   ├── rtm\_client.py  
│   └── static  
│       ├── AgoraRTC\_N-4.24.2.js  
│       ├── agora-rtm-sdk-1.6.0.js  
│       ├── basicRtm.js  
│       ├── basicVideoCall.js  
│       ├── bootstrap.bundle.min.js  
│       ├── bootstrap.min.css  
│       ├── index.css  
│       ├── jquery-3.4.1.min.js  
│       ├── mapbpx-gl.js  
│       └── map.js  
├── guide.md  
├── live\_indoor\_runtime.py  
├── mbra\_repo  
│   ├── deployment  
│   │   ├── LogoNav\_frodobot.py  
│   │   ├── LogoNav\_ros.py  
│   │   ├── model\_weights  
│   │   │   ├── depthest\_ploss  
│   │   │   └── README.md  
│   │   ├── \_\_pycache\_\_  
│   │   │   └── utils\_logonav.cpython-313.pyc  
│   │   ├── utils\_logonav.py  
│   │   └── utils.py  
│   ├── LICENSE  
│   ├── README.md  
│   └── train  
│       ├── config  
│       │   ├── defaults.yaml  
│       │   ├── gnm.yaml  
│       │   ├── late\_fusion.yaml  
│       │   ├── LogoNav.yaml  
│       │   ├── MBRA.yaml  
│       │   ├── nomad.yaml  
│       │   └── vint.yaml  
│       ├── environment\_mbra.yml  
│       ├── mask\_360view.csv  
│       ├── setup.py  
│       ├── train\_environment.yml  
│       ├── train.py  
│       └── vint\_train  
│           ├── data  
│           │   ├── data\_config.yaml  
│           │   ├── data\_utils.py  
│           │   ├── \_\_init\_\_.py  
│           │   ├── lelan\_dataset.py  
│           │   ├── vint\_dataset.py  
│           │   └── vint\_hf\_dataset.py  
│           ├── \_\_init\_\_.py  
│           ├── models  
│           │   ├── base\_model.py  
│           │   ├── depth\_360.py  
│           │   ├── exaug  
│           │   │   ├── exaug.py  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── self\_attention.py  
│           │   │   └── vit.py  
│           │   ├── gnm  
│           │   │   ├── gnm.py  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── modified\_mobilenetv2.py  
│           │   │   ├── \_\_pycache\_\_  
│           │   │   │   ├── gnm.cpython-313.pyc  
│           │   │   │   ├── \_\_init\_\_.cpython-313.pyc  
│           │   │   │   └── modified\_mobilenetv2.cpython-313.pyc  
│           │   │   └── README.md  
│           │   ├── il  
│           │   │   └── il.py  
│           │   ├── \_\_init\_\_.py  
│           │   ├── lelan  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── lelan\_comp.py  
│           │   │   ├── lelan.py  
│           │   │   ├── README.md  
│           │   │   └── sample\_film.py  
│           │   ├── nomad  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── nomad.py  
│           │   │   ├── nomad\_vint.py  
│           │   │   ├── README.md  
│           │   │   └── vib\_placeholder.py  
│           │   ├── \_\_pycache\_\_  
│           │   │   ├── base\_model.cpython-313.pyc  
│           │   │   └── \_\_init\_\_.cpython-313.pyc  
│           │   └── vint  
│           │       ├── \_\_init\_\_.py  
│           │       ├── \_\_pycache\_\_  
│           │       │   ├── \_\_init\_\_.cpython-313.pyc  
│           │       │   └── vint.cpython-313.pyc  
│           │       ├── README.md  
│           │       ├── self\_attention.py  
│           │       ├── vint.py  
│           │       └── vit.py  
│           ├── \_\_pycache\_\_  
│           │   └── \_\_init\_\_.cpython-313.pyc  
│           ├── training  
│           │   ├── logger.py  
│           │   ├── networks  
│           │   │   ├── depth\_decoder\_camera\_ada3.py  
│           │   │   ├── depth\_decoder\_camera\_ada4.py  
│           │   │   ├── depth\_decoder.py  
│           │   │   ├── depth\_decoder\_share.py  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── layers.py  
│           │   │   ├── pose\_cnn.py  
│           │   │   ├── pose\_decoder.py  
│           │   │   └── resnet\_encoder.py  
│           │   ├── train\_eval\_loop.py  
│           │   └── train\_utils.py  
│           └── visualizing  
│               ├── action\_utils.py  
│               ├── distance\_utils.py  
│               ├── \_\_init\_\_.py  
│               └── visualize\_utils.py  
├── mbra\_repo\_1  
│   ├── deployment  
│   │   ├── LogoNav\_frodobot.py  
│   │   ├── LogoNav\_frodobots\_img.py  
│   │   ├── LogoNav\_ros.py  
│   │   ├── test\_logonav\_frodobots\_img.py  
│   │   ├── test\_logonav\_inference.py  
│   │   ├── topological\_graph  
│   │   │   └── data\_collection.py  
│   │   └── utils\_logonav.py  
│   ├── front\_frame.png  
│   ├── LICENSE  
│   ├── README.md  
│   └── train  
│       ├── config  
│       │   ├── defaults.yaml  
│       │   ├── gnm.yaml  
│       │   ├── late\_fusion.yaml  
│       │   ├── LogoNav.yaml  
│       │   ├── MBRA.yaml  
│       │   ├── nomad.yaml  
│       │   └── vint.yaml  
│       ├── data\_split.py  
│       ├── environment\_mbra.yml  
│       ├── lelan.egg-info  
│       │   ├── dependency\_links.txt  
│       │   ├── PKG-INFO  
│       │   ├── SOURCES.txt  
│       │   └── top\_level.txt  
│       ├── mask\_360view.csv  
│       ├── process\_bag\_diff.py  
│       ├── process\_bags.py  
│       ├── process\_recon.py  
│       ├── push\_dataset\_to\_hub.py  
│       ├── setup.py  
│       ├── train\_environment.yml  
│       ├── train.py  
│       └── vint\_train  
│           ├── \_\_init\_\_.py  
│           ├── models  
│           │   ├── base\_model.py  
│           │   ├── depth\_360.py  
│           │   ├── exaug  
│           │   │   ├── exaug.py  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── self\_attention.py  
│           │   │   └── vit.py  
│           │   ├── gnm  
│           │   │   ├── gnm.py  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── modified\_mobilenetv2.py  
│           │   │   └── README.md  
│           │   ├── il  
│           │   │   └── il.py  
│           │   ├── \_\_init\_\_.py  
│           │   ├── lelan  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── lelan\_comp.py  
│           │   │   ├── lelan.py  
│           │   │   ├── README.md  
│           │   │   └── sample\_film.py  
│           │   ├── nomad  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── nomad.py  
│           │   │   ├── nomad\_vint.py  
│           │   │   ├── README.md  
│           │   │   └── vib\_placeholder.py  
│           │   └── vint  
│           │       ├── \_\_init\_\_.py  
│           │       ├── README.md  
│           │       ├── self\_attention.py  
│           │       ├── vint.py  
│           │       └── vit.py  
│           ├── training  
│           │   ├── logger.py  
│           │   ├── networks  
│           │   │   ├── depth\_decoder\_camera\_ada3.py  
│           │   │   ├── depth\_decoder\_camera\_ada4.py  
│           │   │   ├── depth\_decoder.py  
│           │   │   ├── depth\_decoder\_share.py  
│           │   │   ├── \_\_init\_\_.py  
│           │   │   ├── layers.py  
│           │   │   ├── pose\_cnn.py  
│           │   │   ├── pose\_decoder.py  
│           │   │   └── resnet\_encoder.py  
│           │   ├── train\_eval\_loop.py  
│           │   └── train\_utils.py  
│           └── visualizing  
│               ├── action\_utils.py  
│               ├── distance\_utils.py  
│               ├── \_\_init\_\_.py  
│               └── visualize\_utils.py  
├── models  
│   └── README.md  
├── \_\_pycache\_\_  
│   ├── baseline.cpython-313.pyc  
│   └── live\_indoor\_runtime.cpython-313.pyc  
├── README.md  
├── src  
│   ├── corridor\_localizer.py  
│   ├── depth\_estimator.py  
│   ├── depth\_safety.py  
│   ├── earthrover\_interface.py  
│   ├── graph\_planner.py  
│   ├── local\_controller.py  
│   ├── mbra\_controller.py  
│   ├── mbra\_local\_controller.py  
│   ├── navigation\_runtime.py  
│   ├── \_\_pycache\_\_  
│   │   ├── corridor\_localizer.cpython-313.pyc  
│   │   ├── earthrover\_interface.cpython-313.pyc  
│   │   ├── graph\_planner.cpython-313.pyc  
│   │   ├── local\_controller.cpython-313.pyc  
│   │   ├── mbra\_controller.cpython-313.pyc  
│   │   ├── navigation\_runtime.cpython-313.pyc  
│   │   ├── sensor\_state.cpython-313.pyc  
│   │   └── temporal\_localization.cpython-313.pyc  
│   ├── sensor\_state.py  
│   └── temporal\_localization.py  
├── third\_party  
│   └── Depth-Anything-V2  
│       ├── app.py  
│       ├── assets  
│       │   ├── DA-2K.png  
│       │   ├── examples  
│       │   │   ├── demo01.jpg  
│       │   │   ├── demo02.jpg  
│       │   │   ├── demo03.jpg  
│       │   │   ├── demo04.jpg  
│       │   │   ├── demo05.jpg  
│       │   │   ├── demo06.jpg  
│       │   │   ├── demo07.jpg  
│       │   │   ├── demo08.jpg  
│       │   │   ├── demo09.jpg  
│       │   │   ├── demo10.jpg  
│       │   │   ├── demo11.jpg  
│       │   │   ├── demo12.jpg  
│       │   │   ├── demo13.jpg  
│       │   │   ├── demo14.jpg  
│       │   │   ├── demo15.jpg  
│       │   │   ├── demo16.jpg  
│       │   │   ├── demo17.jpg  
│       │   │   ├── demo18.jpg  
│       │   │   ├── demo19.jpg  
│       │   │   └── demo20.jpg  
│       │   ├── examples\_video  
│       │   │   ├── basketball.mp4  
│       │   │   └── ferris\_wheel.mp4  
│       │   └── teaser.png  
│       ├── DA-2K.md  
│       ├── depth\_anything\_v2  
│       │   ├── dinov2\_layers  
│       │   │   ├── attention.py  
│       │   │   ├── block.py  
│       │   │   ├── drop\_path.py  
│       │   │   ├── \_\_init\_\_.py  
│       │   │   ├── layer\_scale.py  
│       │   │   ├── mlp.py  
│       │   │   ├── patch\_embed.py  
│       │   │   └── swiglu\_ffn.py  
│       │   ├── dinov2.py  
│       │   ├── dpt.py  
│       │   └── util  
│       │       ├── blocks.py  
│       │       └── transform.py  
│       ├── LICENSE  
│       ├── metric\_depth  
│       │   ├── assets  
│       │   │   └── compare\_zoedepth.png  
│       │   ├── dataset  
│       │   │   ├── hypersim.py  
│       │   │   ├── kitti.py  
│       │   │   ├── splits  
│       │   │   │   ├── hypersim  
│       │   │   │   │   ├── train.txt  
│       │   │   │   │   └── val.txt  
│       │   │   │   ├── kitti  
│       │   │   │   │   └── val.txt  
│       │   │   │   └── vkitti2  
│       │   │   │       └── train.txt  
│       │   │   ├── transform.py  
│       │   │   └── vkitti2.py  
│       │   ├── depth\_anything\_v2  
│       │   │   ├── dinov2\_layers  
│       │   │   │   ├── attention.py  
│       │   │   │   ├── block.py  
│       │   │   │   ├── drop\_path.py  
│       │   │   │   ├── \_\_init\_\_.py  
│       │   │   │   ├── layer\_scale.py  
│       │   │   │   ├── mlp.py  
│       │   │   │   ├── patch\_embed.py  
│       │   │   │   └── swiglu\_ffn.py  
│       │   │   ├── dinov2.py  
│       │   │   ├── dpt.py  
│       │   │   └── util  
│       │   │       ├── blocks.py  
│       │   │       └── transform.py  
│       │   ├── depth\_to\_pointcloud.py  
│       │   ├── dist\_train.sh  
│       │   ├── README.md  
│       │   ├── requirements.txt  
│       │   ├── run.py  
│       │   ├── train.py  
│       │   └── util  
│       │       ├── dist\_helper.py  
│       │       ├── loss.py  
│       │       ├── metric.py  
│       │       └── utils.py  
│       ├── README.md  
│       ├── requirements.txt  
│       ├── run.py  
│       └── run\_video.py  
├── tools  
│   ├── evaluate\_temporal\_localization.py  
│   ├── extract\_h5\_dataset.py  
│   └── \_\_pycache\_\_  
│       ├── evaluate\_temporal\_localization.cpython-313.pyc  
│       └── extract\_h5\_dataset.cpython-313.pyc  
└── verify\_workspace.py

77 directories, 2189 files  
(base) vivek@pop-os:\~/Desktop/rover/ERC-3-earthrover-challenge$ ^C  
(base) vivek@pop-os:\~/Desktop/rover/ERC-3-earthrover-challenge$ ls  
baseline.py                    earth-rovers-sdk        \_\_pycache\_\_  
chat\_handoff.md                guide.md                README.md  
chat\_session\_reconstructed.md  live\_indoor\_runtime.py  src  
CONTEXT.md                     mbra\_repo               third\_party  
data                           mbra\_repo\_1             tools  
docs                           models                  verify\_workspace.py  
(base) vivek@pop-os:\~/Desktop/rover/ERC-3-earthrover-challenge$ 

