const video = document.getElementById("video");
const gameId = document.getElementById("gameId").value;

async function startCam()
{
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    })

    const vidCanvas = document.createElement("canvas");
    const ctx = vidCanvas.getContext("2d");

    video.addEventListener("loadeddata", () => {
        vidCanvas.width = video.videoWidth;
        vidCanvas.height = video.videoHeight;

        setInterval(() => {
            ctx.drawImage(video, 0, 0, vidCanvas.width, vidCanvas.height);
            const image = vidCanvas.toDataURL("image/jpeg");

            fetch(`/api/game/${gameId}/frame`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({image: image})
            });
            console.log("sucessfully fetched")
        }, 100);
    });

    video.srcObject = stream;
    await video.play();
}

startCam();