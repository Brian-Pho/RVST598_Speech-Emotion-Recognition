// Standard/global options
var options = {
    responsive: false,
    // animation: {
    //     onComplete: function(animation){
    //         document.querySelector('.savegraph').setAttribute('href', this.toBase64Image());
    //     }
    // }
}
// Standard size
const WIDTH = 500;
const HEIGHT = 500;
function resizeCanvas(canvas) {
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style = "margin:auto;"
}
Chart.helpers.merge(Chart.defaults.global.plugins.datalabels, {
    color: 'white'
});
// Standard colors
window.chartColors = {
	red: 'rgb(255, 99, 132)',
	orange: 'rgb(255, 159, 64)',
	yellow: 'rgb(255, 205, 86)',
	green: 'rgb(75, 192, 192)',
	blue: 'rgb(54, 162, 235)',
	purple: 'rgb(153, 102, 255)',
	grey: 'rgb(231,233,237)'
};

// Number of male and female samples
var mFPC = document.getElementById('maleFemalePieChart');
resizeCanvas(mFPC)
var myDoughnutChart = new Chart(mFPC, {
    type: 'doughnut',
    data: {
        datasets: [{
            data: [50, 50],
            backgroundColor: [window.chartColors.red, window.chartColors.blue]
        }],
    
        // These labels appear in the legend and in the tooltips when hovering different arcs
        labels: ['Female', 'Male']
    },
    options: Object.assign(options, {
        title: {
            display: true,
            text: 'Genders'
        }
    })
});

// Number of samples in each emotion class
var eCPC = document.getElementById('emotionClassPieChart');
resizeCanvas(eCPC)
var myDoughnutChart = new Chart(eCPC, {
    type: 'doughnut',
    data: {
        datasets: [{
            data: [5994, 6209, 1960, 1884, 4745, 3040, 883],
            backgroundColor: [window.chartColors.red, window.chartColors.orange, window.chartColors.yellow,
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
            data: [19884, 727, 1064, 45, 1],
            backgroundColor: [window.chartColors.red, window.chartColors.orange, window.chartColors.yellow,
            window.chartColors.green,
            window.chartColors.blue]
        }],
    
        // These labels appear in the legend and in the tooltips when hovering different arcs
        labels: ["1", "2", "3", "4", "5"]
    },
    options: Object.assign(options, {
        title: {
            display: true,
            text: 'Label Types'
        }
    })
});

// Number of samples from each database