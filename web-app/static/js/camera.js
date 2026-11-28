const video = document.getElementById("video");
const gameId = document.getElementById("gameId").value;

let countdown = 3;
let time = 10;
const leftcountdownDisplay = document.querySelector(".left-countdown");
const rightcountdownDisplay = document.querySelector(".right-countdown");
const timeDisplay = document.querySelector(".time")

async function startCam()
{
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    })

    const vidCanvas = document.createElement("canvas");
    const ctx = vidCanvas.getContext("2d");

    video.srcObject = stream;
    await video.play();

    vidCanvas.width = video.videoWidth;
    vidCanvas.height = video.videoHeight;

    leftcountdownDisplay.style.visibility = "visible";
    rightcountdownDisplay.style.visibility = "visible";

    const countInterval = setInterval(() => {
        countdown--;
        if (countdown > 0)
        {
            leftcountdownDisplay.textContent = countdown;
            rightcountdownDisplay.textContent = countdown;
        }
        else if (countdown === 0)
        {
            leftcountdownDisplay.textContent = "GO!";
            rightcountdownDisplay.textContent = "GO!";

            const timeInterval = setInterval(() => {
                time --;
                if (time > 0)
                {
                    timeDisplay.textContent = `0:${String(time).padStart(2, "0")}`;
                }
                else
                {
                    clearInterval(timeInterval);
                    window.location.href = `/api/game/${gameId}/finish`;
                }
            }, 1000);

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
        }

        else
        {
            clearInterval(countInterval);
            leftcountdownDisplay.style.visibility = "hidden";
            rightcountdownDisplay.style.visibility = "hidden";
        }
    }, 1000);
}

startCam();