// Standard/global options
var options = {
    responsive: false,
}

// Standard size
const WIDTH = 900;
const HEIGHT = 900;
function resizeCanvas(canvas) {
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style = "margin:auto;"
}
Chart.helpers.merge(Chart.defaults.global.plugins.datalabels, {
    color: 'white'
});

// Set font size
Chart.defaults.global.defaultFontSize = 20;

// Standard colors
window.chartColors = {
	red: 'rgb(255, 99, 132)',
	orange: 'rgb(255, 159, 64)',
	yellow: 'rgb(255, 205, 86)',
	green: 'rgb(75, 192, 192)',
	blue: 'rgb(54, 162, 235)',
	purple: 'rgb(153, 102, 255)',
	grey: 'rgb(148, 150, 153)'
};

// Number of samples in each emotion class
var eCPC = document.getElementById('trainEmotionClassPieChart');
resizeCanvas(eCPC)
var myDoughnutChart = new Chart(eCPC, {
    type: 'doughnut',
    data: {
        datasets: [{
            data: [5954, 6177, 1934, 1851, 4732, 3006, 876],
            backgroundColor: [
                window.chartColors.red, 
                window.chartColors.orange, 
                window.chartColors.yellow,
                window.chartColors.green,
                window.chartColors.blue,
                window.chartColors.purple,
                window.chartColors.grey]
        }],
    
        // These labels appear in the legend and in the tooltips when hovering different arcs
        labels: ["Neutral", "Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    },
    options: Object.assign(options, {
        title: {
            display: true,
            text: 'Emotions'
        }
    })
});

// Number of samples in each label type
var lTPC = document.getElementById('labelTypePieChart');
resizeCanvas(lTPC)
var myDoughnutChart = new Chart(lTPC, {
    type: 'doughnut',
    data: {
        datasets: [{
            data: [19884, 727, 1064],
            backgroundColor: [
                window.chartColors.red, 
                window.chartColors.orange, 
                window.chartColors.yellow,
                window.chartColors.green,
                window.chartColors.blue]
        }],
    
        // These labels appear in the legend and in the tooltips when hovering different arcs
        labels: ["One", "Two", "Three"]
    },
    options: Object.assign(options, {
        title: {
            display: true,
            text: 'Multi-Label Types'
        }
    })
});

// Number of samples from each database
// Number of samples in each label type
var dOPC = document.getElementById('databaseOriginPieChart');
resizeCanvas(dOPC)
var myDoughnutChart = new Chart(dOPC, {
    type: 'doughnut',
    data: {
        datasets: [{
            data: [7414, 10021, 1440, 2800],
            backgroundColor: [
                window.chartColors.red, 
                window.chartColors.orange, 
                window.chartColors.yellow,
                window.chartColors.green]
        }],
    
        // These labels appear in the legend and in the tooltips when hovering different arcs
        labels: ["CREMA-D", "IEMOCAP", "RAVDESS", "TESS"]
    },
    options: Object.assign(options, {
        title: {
            display: true,
            text: 'Samples per Database'
        }
    })
});