class Stats {

    constructor() {
        this.lastFrameTime = performance.now();
        this.currentFrameTime = performance.now();
        this.beginRunTime = performance.now();
        this.endRunTime = performance.now();
        this.elapsedTime = 0;
        //this.frameCount = 0;
        //this.frameCountSinceBegin = 0;
        this.fps = 0;
    }

    begin() {
        this.beginRunTime = performance.now();
    }

    end() {
        this.endRunTime = performance.now();
    }

    updateFPS() {
        //this.frameCount++;
        //this.frameCountSinceBegin++;

        this.currentFrameTime = performance.now();
        this.elapsedTime = this.currentFrameTime - this.lastFrameTime;
        this.fps = 1000.0 / this.elapsedTime;
        //this.fps = 1000.0 / elapsedTime;
        //this.frameCount = 0;
        this.lastFrameTime = this.currentFrameTime;
    }

    get FPS() {
        return this.fps.toFixed(1);
    }

    get ElapsedTime() {
        return this.elapsedTime.toFixed(1);
    }

    get RunTime() {
        return (this.endRunTime - this.beginRunTime);
    }

}

export {Stats}