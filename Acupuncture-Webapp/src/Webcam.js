export class Webcam {
    constructor(webcamElement, canvasElement) {
      this.webcamElement = webcamElement;
      this.canvasElement = canvasElement;
    }

    adjustVideoSize(width, height) {
      const aspectRatio = width / height;
      if (width >= height) {
          this.webcamElement.width = aspectRatio * this.webcamElement.height;
      } else  {
          this.webcamElement.height = this.webcamElement.width / aspectRatio;
      }
    }

    async setup() {
        return new Promise((resolve, reject) => {
          if (navigator.mediaDevices.getUserMedia !== undefined) {
            navigator.mediaDevices.getUserMedia({
                audio: false, video: { facingMode: 'user' }
                })
                .then((mediaStream) => {
                    if ("srcObject" in this.webcamElement) {
                        this.webcamElement.srcObject = mediaStream;
                    } else {
                        // For older browsers without the srcObject.
                        this.webcamElement.src = window.URL.createObjectURL(mediaStream);
                    }
                    this.webcamElement.addEventListener(
                        'loadeddata',
                        async () => {
                            this.adjustVideoSize(
                                this.webcamElement.videoWidth,
                                this.webcamElement.videoHeight
                            );
                            resolve();
                        },
                        false
                    );
                });
          } else {
              reject();
          }
      });
    }

}