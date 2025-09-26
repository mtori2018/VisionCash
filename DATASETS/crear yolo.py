pip install "ultralytics==8.2.103" ^
            "tensorflow==2.15.0.post1" ^
            "keras==2.15.0" ^
            "onnx==1.14.0" "onnxsim==0.4.33" "onnx2tf==1.19.13"
# activa Keras legacy dentro de TF
$env:TF_USE_LEGACY_KERAS="1"
