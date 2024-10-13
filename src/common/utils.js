import {Keypoint, Vector2D} from "./types.js";

const jointNames = ['rightShoulder', 'rightElbow', 'rightWrist', 'leftShoulder',
    'leftElbow', 'leftWrist', 'rightHip', 'rightKnee',
    'rightAnkle', 'leftHip', 'leftKnee', 'leftAnkle',
    'headTop', 'upperNeck', 'leftEye', 'rightEye'];

const jointIndices = jointNames.reduce((indices, jointName, curIndex) => {
    indices[jointName] = curIndex;
    return indices;
}, {});

const jointPairNames = [
    ['leftEye', 'rightEye'], ['headTop', 'upperNeck'], ['leftShoulder', 'rightShoulder'],
    ['leftShoulder', 'leftHip'], ['rightShoulder', 'rightHip'], ['leftHip', 'rightHip'],
    ['leftHip', 'leftKnee'], ['leftKnee', 'leftAnkle'], ['rightHip', 'rightKnee'],
    ['rightKnee', 'rightAnkle'], ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
    ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist']
];

const jointPairIndices = jointPairNames.map(([jointNameFrom, jointNameTo]) => ([jointIndices[jointNameFrom], jointIndices[jointNameTo]]));

function getValidKeypointIndexPairs(keypoints, minConfidence) {
    return jointPairIndices.reduce((validPairs, [jointFromIndex, jointToIndex]) => {
        if (keypoints[jointFromIndex].score < minConfidence || keypoints[jointToIndex].score < minConfidence) {
            return validPairs;
        } else {
            validPairs.push([jointFromIndex, jointToIndex]);
        }
        return validPairs;
    }, [])
}

function drawPoint(ctx, x, y, point_radius, color) {
    ctx.beginPath();
    ctx.arc(x, y, point_radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawSegment(ctx, start_point, end_point, color, lineWidth = 2, scale = 1) {
    ctx.beginPath();
    ctx.moveTo(start_point.x * scale, start_point.y * scale);
    ctx.lineTo(end_point.x * scale, end_point.y * scale);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();
}

function drawSkeleton(ctx, keypoints, minConfidence, color = 'red', lineWidth = 2, scale = 1) {
    const keypointIndexPairs = getValidKeypointIndexPairs(keypoints, minConfidence);
    keypointIndexPairs.forEach(([from, to]) => {
        drawSegment(ctx, keypoints[from].position, keypoints[to].position, color, lineWidth, scale);
    });
}

function drawRect(ctx, rect, color = 'red', lineWidth = 2, scale = 1) {
    let left_top = new Vector2D(rect.left, rect.top);
    let left_bottom = new Vector2D(rect.left, rect.top + rect.height);
    let right_top = new Vector2D(rect.left + rect.width, rect.top);
    let right_bottom = new Vector2D(rect.left + rect.width, rect.top + rect.height);

    drawSegment(ctx, left_top, left_bottom, color, lineWidth, scale);
    drawSegment(ctx, left_top, right_top, color, lineWidth, scale);
    drawSegment(ctx, left_bottom, right_bottom, color, lineWidth, scale);
    drawSegment(ctx, right_top, right_bottom, color, lineWidth, scale);

    // drawPoint(ctx, left_top.x, left_top.y, 5, color);
    // drawPoint(ctx, left_bottom.x, left_bottom.y, 5, color);
    // drawPoint(ctx, right_top.x, right_top.y, 5, color);
    // drawPoint(ctx, right_bottom.x, right_bottom.y, 5, color);
}

function drawKeypoints(ctx, keypoints, minConfidence, point_radius, color, scale = 1) {
    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];

        if (keypoint.score < minConfidence) {
            continue;
        }

        const {x, y} = keypoint.position;
        drawPoint(ctx, x * scale, y * scale, point_radius, color);
    }
}

function checkGLError(gl, op) {
    let error;
    while ((error = gl.getError()) !== gl.NO_ERROR) {
        console.error("operation:" + op + " glError: 0x" + error.toString(16));
    }

}

function getBoundingBox(pixel_x, pixel_y, radius, width, height) {
    let ymin = Math.max(pixel_y - radius, 0);
    let ymax = Math.min(pixel_y + radius, height - 1);
    let xmin = Math.max(pixel_x - radius, 0);
    let xmax = Math.min(pixel_x + radius, width - 1);
    return {xmin, xmax, ymin, ymax};
}

function findCandidateKeypoints(heatmaps, max_thresh = 0.039, wh_radius, width, height, channel) {
    //net output [16x64x64]->c, h, w
    let pose = [];
    let act_thresh = 0.0002;
    for (let c = 0; c < channel; c++) {
        let channel_offset = c * height * width;
        let max_confidence = 0;
        let avg_confidence = 0;
        let cur_width_index = 0;
        let cur_height_index = 0;

        for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
                let index = h * width + w;
                let confidence = heatmaps[channel_offset + index];
                avg_confidence += confidence;
                if (confidence > max_confidence) {
                    max_confidence = confidence;
                    cur_width_index = w;
                    cur_height_index = h;
                }
            }
        }
        avg_confidence /= (width*height);
        // let data_wh = heatmaps.slice(channel_offset, channel_offset + width * height);
        // max_confidence = Math.max(...data_wh);
        // let max_index = data_wh.indexOf(max_confidence);
        // cur_height_index = parseInt(max_index / width);
        // cur_width_index = max_index % width;

        if (avg_confidence > act_thresh && max_confidence >= max_thresh) {
            let sum_confidence = 0;
            //caculate gravity center
            let {xmin, xmax, ymin, ymax} = getBoundingBox(cur_width_index, cur_height_index, wh_radius, width, height);
            let sum_x = 0;
            let sum_y = 0;
            for (let h = ymin; h <= ymax; h++) {
                let sum_line_x = 0;
                for (let w = xmin; w <= xmax; w++) {
                    let index = h * width + w;
                    let confidence = heatmaps[channel_offset + index];
                    sum_confidence += confidence;
                    sum_x += w * confidence;
                    sum_line_x += confidence;
                }
                sum_y += h * sum_line_x;
            }

            let center_x = Math.round(sum_x / sum_confidence);
            let center_y = Math.round(sum_y / sum_confidence);

            pose[c] = new Keypoint(new Vector2D(center_x, center_y), max_confidence);
        } else {
            pose[c] = new Keypoint(new Vector2D(-1, -1), 0);
        }
    }

    return pose;

}

function compareFeatureFile (feature_file0,feature_file1){
    const promises = Array.of(getJSON(feature_file0),getJSON(feature_file1));

    Promise.all(promises).then(function (jsons) {
        let feature_data_out = new Array(jsons.length);
        jsons.forEach(function (feature_data, index) {
            let arr = Object.keys(feature_data);
            let data = new Float32Array(arr.length);
            for (let key in feature_data) {
                data[parseInt(key)] = parseFloat(feature_data[key]);
            }
            feature_data_out[index] = data;
        });

        console.assert(feature_data_out[0].length == feature_data_out[1].length, 'feature/run data length mismatch');
        let max_error = 0;
        let avg_error = 0;
        let sum_error = 0;
        let max_error_index = 0;
        let error_num = 0;

        feature_data_out[1].forEach(function (value, index) {
            let error = Math.abs(value - feature_data_out[0][index]);
            if (error >= 0.000001) {
                error_num++;
            }
            sum_error += error;
            if (error > max_error) {
                max_error = error;
                max_error_index = index;
            }
        });

        avg_error = sum_error / feature_data_out[1].length;

        let test_result = 'test' + ((max_error >= 0.001) ? ' fail' : ' pass');

        console.log(test_result, 'avg:', avg_error.toFixed(10), ', max:', max_error.toFixed(10), 'at (', max_error_index, '), errors:', error_num, "(error >= 0.000001)");
    }).catch(function (error) {
        console.error(error);
    });
}

const getJSON = function (url) {
    const promise = new Promise(function (resolve, reject) {
        const handler = function () {
            if (this.readyState !== 4) {
                return;
            }
            if (this.status === 200) {
                resolve(this.response);
            } else {
                reject(new Error(this.statusText));
            }
        };
        const client = new XMLHttpRequest();
        client.open("GET", url);
        client.onreadystatechange = handler;
        client.responseType = "json";
        client.setRequestHeader("Accept", "application/json");
        client.send();

    });

    return promise;
};

function checkBrowerSupport() {
    if (!window.requestAnimationFrame) {
        window.requestAnimationFrame = (function () {
            return window.requestAnimationFrame ||
                window.webkitRequestAnimationFrame ||
                window.mozRequestAnimationFrame ||
                window.oRequestAnimationFrame ||
                window.msRequestAnimationFrame ||
                function (callback, element) {
                    window.setTimeout(callback, 1000 / 60);
                };
        })();
    }

    if (!window.cancelAnimationFrame) {
        window.cancelAnimationFrame = (window.cancelRequestAnimationFrame ||
            window.webkitCancelAnimationFrame || window.webkitCancelRequestAnimationFrame ||
            window.mozCancelAnimationFrame || window.mozCancelRequestAnimationFrame ||
            window.msCancelAnimationFrame || window.msCancelRequestAnimationFrame ||
            window.oCancelAnimationFrame || window.oCancelRequestAnimationFrame ||
            window.clearTimeout);
    }
}

function checkWebGL() {
    let canvas = document.createElement('canvas');
    let info = document.getElementById('info');

    let gl = canvas.getContext('webgl2', {antialias: false});

    if (!gl) {
        let gl = canvas.getContext('webgl', {antialias: false}) || canvas.getContext('experimental-webgl', {antialias: false});
        if (!gl) {
            info.textContent = 'This browser does not support webgl.';
            info.style.display = 'block';
            //alert('This browser does not support webgl.');
        }
        info.textContent = 'This browser does not support webgl2.';
        info.style.display = 'block';
        //alert('This browser does not support webgl2.');
        return;
    }

    let ext = gl.getExtension('EXT_color_buffer_float');

    if (!ext) {
        info.textContent = 'Warning: Float texture is not supported.';
        info.style.display = 'block';
        //alert('Warning: Float texture is not supported.');
        return;
    }

    return gl;
}

function isAndroid() {
    return /Android/i.test(navigator.userAgent);
}

function isIOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
    return isAndroid() || isIOS();
}

async function setupCamera(videoWidth, videoHeight) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;

    const mobile = isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: mobile ? videoWidth : videoWidth,
            height: mobile ? videoWidth : videoHeight,
        },
    });

    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo(videoWidth, videoHeight) {
    const video = await setupCamera(videoWidth, videoHeight);
    video.play();
    return video;
}


(function (console) {

    // console.save = function(data, filename){
    //
    //     if(!data) {
    //         console.error('No data');
    //         return;
    //     }
    //     if(!filename) filename = 'console.json';
    //
    //     if(typeof data === 'object'){
    //         data = JSON.stringify(data, undefined, 4);
    //     }
    //
    //     let blob = new Blob([data], {type: 'text/json'});
    //     let ev = document.createEvent('MouseEvents');
    //     let down = document.createElement('a');
    //     down.download = filename;
    //     down.href = window.URL.createObjectURL(blob);
    //     down.dataset.downloadurl = ['text/json', down.download, down.href].join(':');
    //     ev.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
    //     down.dispatchEvent(ev);
    //
    // }

    console.save = function (content, filename) {
        if (!content) {
            console.error('No input content');
            return;
        }

        if (!filename) filename = 'console.json';

        let eleLink = document.createElement('a');
        eleLink.download = filename;
        eleLink.style.display = 'none';

        if (typeof (content) === 'object') {
            content = JSON.stringify(content, undefined, 4);
        }
        let blob = new Blob([content], {type: 'text/json'});
        eleLink.href = URL.createObjectURL(blob);
        document.body.appendChild(eleLink);
        eleLink.click();
        document.body.removeChild(eleLink);
    };

})(console)

export {drawPoint, drawKeypoints, drawSegment, drawSkeleton, drawRect, checkGLError, getBoundingBox, findCandidateKeypoints, compareFeatureFile, getJSON, checkBrowerSupport, checkWebGL,
        isAndroid, isIOS, isMobile, setupCamera, loadVideo}

