import {selection} from d3;
// Function to upload the fil
console.log(selection)
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];  // The selected file

    if (file) {
        const formData = new FormData();
        formData.append('file', file);  // Pack file into FormData

        // Send file to backend using fetch
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Successfully uploaded:', data);
            alert('File successfully uploaded!');

            // Populate dropdowns with options from backend response
            populateDropdowns(data);
        })
        .catch(error => {
            console.error('Error uploading:', error);
            alert('Error uploading the file.');
        });
    } else {
        alert('Please select a file.');
    }
}

// Fill dropdowns with backend data
function populateDropdowns(options) {
    const dropdown1 = document.getElementById('dropdown1');
    const dropdown2 = document.getElementById('dropdown2');

    // Reset dropdowns
    dropdown1.innerHTML = '<option value="">Select an option</option>';
    dropdown2.innerHTML = '<option value="">Select an option</option>';

    options.forEach(option => {
        // First dropdown
        const optionElement1 = document.createElement('option');
        optionElement1.value = option;
        optionElement1.textContent = option;
        dropdown1.appendChild(optionElement1);

        // Second dropdown
        const optionElement2 = document.createElement('option');
        optionElement2.value = option;
        optionElement2.textContent = option;
        dropdown2.appendChild(optionElement2);
    });
}

// Submit selections to backend
function submitSelections() {
    const dropdown1 = document.getElementById('dropdown1').value;
    const dropdown2 = document.getElementById('dropdown2').value;
    const analysisOption = document.getElementById('analysisDropdown').value;
    const batchType = document.getElementById('batchTypeDropdown').value;
    const gamma = document.getElementById('gammaInput').value;

    if (dropdown1 && dropdown2 && analysisOption && batchType && gamma) {
        fetch('/process_selections', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                selection1: dropdown1,
                selection2: dropdown2,
                analysis: analysisOption,
                batchType: batchType,
                gamma: parseFloat(gamma) // Convert gamma to a number
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                console.log('Server response:', data.data);
                displayPerformanceSpectrum(data.data);  // Call D3 visualization function
            }
        })
        .catch(error => {
            console.error('Error sending selection:', error);
            alert('Error sending selection.');
        });
    } else {
        alert('Please select an option in all dropdowns and enter a gamma value.');
    }
}

function fetchAndDisplayData() {
    fetch('/process_selections', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not OK");
        }
        return response.json();  // Parse the response as JSON
    })
    .then(data => {
        console.log("JSON response from server:", data);  // Print the JSON to the console

        // Here you can add your code to process/display the data
        displayPerformanceSpectrum(data);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
    });
}

function displayPerformanceSpectrum(data) {
    const margin = {top: 20, right: 20, bottom: 30, left: 50};
    const width = 1000 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    // Parse the time and duration data
    const startTimes = data.map(d => d.start_time);
    const endTimes = data.map(d => d.end_time);
    const durations = data.map(d => d.duration);

    // Set up scales for start time and end time
    const xStart = d3.scaleTime()
        .domain([d3.min(startTimes), d3.max(startTimes)])
        .range([0, width]);

    const xEnd = d3.scaleTime()
        .domain([d3.min(endTimes), d3.max(endTimes)])
        .range([0, width]);

    const yScale = d3.scaleBand()
        .domain(data.map(d => d.case_ID))
        .range([0, height])
        .padding(0.1);

    // Set up the SVG container
    const svg = d3.select("#chart")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Draw start and end time lines for each case
    svg.selectAll(".case-line")
        .data(data)
        .enter()
        .append("line")
        .attr("class", "case-line")
        .attr("x1", d => xStart(d.start_time))
        .attr("x2", d => xEnd(d.end_time))
        .attr("y1", d => yScale(d.case_ID) + yScale.bandwidth() / 2)
        .attr("y2", d => yScale(d.case_ID) + yScale.bandwidth() / 2)
        .attr("stroke", "steelblue")
        .attr("stroke-width", 2);

    // Add axes
    const xAxisStart = d3.axisBottom(xStart).ticks(10).tickFormat(d3.timeFormat("%Y-%m-%d"));
    const xAxisEnd = d3.axisTop(xEnd).ticks(10).tickFormat(d3.timeFormat("%Y-%m-%d"));

    svg.append("g")
        .attr("class", "x-axis-start")
        .attr("transform", `translate(0,${height + margin.bottom})`)
        .call(xAxisStart);

    svg.append("g")
        .attr("class", "x-axis-end")
        .call(xAxisEnd);

    // Add y-axis (case IDs)
    svg.append("g")
        .attr("class", "y-axis")
        .call(d3.axisLeft(yScale));
}

// Trigger the data fetch and visualization on page load or button click
fetchAndDisplayData();
