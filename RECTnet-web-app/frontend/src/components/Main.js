import React, { PureComponent } from 'react';
import axios from 'axios';

import { makeStyles } from '@material-ui/core/styles';
import CssBaseline from '@material-ui/core/CssBaseline';
import Drawer from '@material-ui/core/Drawer';
import Box from '@material-ui/core/Box';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import List from '@material-ui/core/List';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import IconButton from '@material-ui/core/IconButton';
import Badge from '@material-ui/core/Badge';
import Container from '@material-ui/core/Container';
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import Link from '@material-ui/core/Link';
import MenuIcon from '@material-ui/icons/Menu';
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft';
import NotificationsIcon from '@material-ui/icons/Notifications';
import { mainListItems, secondaryListItems } from './listItems';
import Button from '@material-ui/core/Button';
import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import Loader from 'react-loader-spinner';
import { PieChart, Pie, Sector } from 'recharts';
import cloneDeep from 'lodash/cloneDeep';
import '../styles/Main.css';

/*
Refactor:
1) Put all in-line CSS into CSS failes
2) Put pie chart functions into separate component file
3) For resetting state.data, remove lines 154-168, create new function
4) lines 185-189, think of better way for accessing nested JSON
*/


// Pie chart used to display predicted emotions
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];
const RADIAN = Math.PI / 180;

const renderActiveShape = (props) => {
  const {
    cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle,
    fill, payload, percent, value,
  } = props;
  const sin = Math.sin(-RADIAN * midAngle);
  const cos = Math.cos(-RADIAN * midAngle);
  const sx = cx + (outerRadius + 10) * cos;
  const sy = cy + (outerRadius + 10) * sin;
  const mx = cx + (outerRadius + 30) * cos;
  const my = cy + (outerRadius + 30) * sin;
  const ex = mx + (cos >= 0 ? 1 : -1) * 22;
  const ey = my;
  const textAnchor = cos >= 0 ? 'start' : 'end';

  return (
    <g>
      <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill}>Emotions</text>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 6}
        outerRadius={outerRadius + 10}
        fill={fill}
      />
      <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
      <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
      <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="#333">{`${payload.name}`}</text>
      <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} dy={18} textAnchor={textAnchor} fill="#999">
        {`(Rate ${(percent * 100).toFixed(2)}%)`}
      </text>
    </g>
  );
};

class Main extends React.Component {
  static jsfiddleUrl = 'https://jsfiddle.net/alidingling/hqnrgxpj/';
  constructor(props) {
    super(props);

    this.state = {
      selectedFile: '',
      imagePreviewUrl: null,
      // modelPredictionText: '',
      predictedResult: '',
      resultFinishedLoading: '',

      // Used for pie chart
      activeIndex: 0,
      height: 400,
      data: [
        { name: 'Neutral',  value: 0.00 },
        { name: 'Happiness',    value: 0.00 },
        { name: 'Sadness',    value: 0.00 },
        { name: 'Surprise',   value: 0.00 },
        { name: 'Fear',       value: 0.00 },
        { name: 'Disgust',    value: 0.00 },
        { name: 'Anger',      value: 0.00 },        
      ]
    };
  }

  // Used for pie chart
  componentDidMount() {
    const height = 400;
    if (this.divElement && this.divElement.clientHeight) {
      height = this.divElement.clientHeight;
    }
    this.setState({ height });
  }

  onPieEnter = (data, index) => {
    this.setState({
      activeIndex: index,
    });
  };

  // Used for uploading photo
  fileSelectedHandler = event => {
    event.preventDefault();
    let reader = new FileReader();
    let file = event.target.files[0]

    reader.onloadend = () => {
        this.setState({
            selectedFile: file,
            imagePreviewUrl: reader.result
        })
    }
    reader.readAsDataURL(file);
  }

  predictHandler = (event) => {    
    if (this.state.selectedFile !== '') {
      this.setState({resultFinishedLoading: false}); 

      const fd = new FormData();
      fd.append('file',  this.state.selectedFile);

      fetch (
      '/predict', {
          method: 'POST',
          body: fd
      })
      .then((response) => {
        return response.json();
      })
      .then((data) => {

        let emptyData = [
          { name: 'Neutral',  value: 0.00 },
          { name: 'Happiness',    value: 0.00 },
          { name: 'Sadness',    value: 0.00 },
          { name: 'Surprise',   value: 0.00 },
          { name: 'Fear',       value: 0.00 },
          { name: 'Disgust',    value: 0.00 },
          { name: 'Anger',      value: 0.00 },        
        ]

        // Reset the state after a new photo is uploaded
        // So that previous prediction doesn't affect current prediction
        this.setState({
          data: emptyData
        })

        // Change incoming data to correct JSON format
        data = data.split('(').join('[');
        data = data.split(')').join(']');
        data = data.replace(/\'/g, '\"');
        
        let result = JSON.parse(data);
        let primaryMood = '';

        this.setState({
          predictedResult: result,
          resultFinishedLoading: true
        })

        if (this.state.predictedResult && this.state.predictedResult['faces'] && this.state.predictedResult['faces'][0]
                                                                          && this.state.predictedResult['faces'][0]['class']) {
          primaryMood = this.state.predictedResult['faces'][0]['class'].split(" ")[1];

          let dataCopy = cloneDeep(this.state.data);

          for (let i = 0; i <  dataCopy.length; i++) {
            if (dataCopy[i].name === primaryMood) {
              this.setState({activeIndex: i});
            }
            dataCopy[i].value = this.state.predictedResult['faces'][0]['details'][i];
          }
  
          this.setState({data: dataCopy});

          // NOTE Save to history 
          // Only if there is one face in the picture
          if (this.state.predictedResult['faces'].length == 1) {
            let hData = []
            this.state.predictedResult['faces'][0].details.forEach(v=>hData.push(parseInt(100000*v)/1000))
            if (!!this.props.saveHistory) {
              this.props.saveHistory(hData)
            }
          }

        } else {
          console.log("No faces detected in the photo");
        }
      });      
    }
  }

  render() {
    let dim = this.state.height / 1.2;

    let imageStyle = {
        marginTop: 50,
        marginLeft: 10,
    }

    const useStyles = makeStyles((theme) => ({
      root: {
        '& > *': {
          margin: theme.spacing(1),
        },
      },
      input: {
        display: 'none',
      },
    }));

    // Display the prediction result
    // If it hasn't finished loading yet, display the spinner
    let predictionResult;
    let pieChartComponent;

    if (this.state.resultFinishedLoading === true) {
      if (this.state.predictedResult && this.state.predictedResult['faces'] && this.state.predictedResult['faces'][0]) {
            predictionResult = <div className="PredictResult">
                                  <p style={{marginTop: 70 ,fontFamily: 'sans-serif', fontSize: 20, color: '#3f51b5'}}>Prediction:   
                                  {'   '}
                                  {this.state.predictedResult['faces'][0]['class'].split(" ")[1]}
                                  </p>
                                </div>;

            pieChartComponent =  <div className="PredictResult"
                                    ref={ (divElement) => { this.divElement = divElement } }>
                                        <PieChart width={800} height={400}>
                                            <Pie
                                                activeIndex={this.state.activeIndex}
                                                activeShape={renderActiveShape}
                                                data={this.state.data}
                                                cx={dim/2 + 80}
                                                cy={dim/2}
                                                innerRadius={dim/5 + 40}
                                                outerRadius={dim/4 + 40}
                                                fill="#8884d8"
                                                dataKey="value"
                                                onMouseEnter={this.onPieEnter}
                                            />
                                        </PieChart>
                                    </div>
      } else {
        predictionResult = <div className="PredictResult">
                            <p style={{marginTop: 70 ,fontFamily: 'sans-serif', fontSize: 20, color: '#3f51b5'}}>No Face(s) detected.</p>
                          </div>;
        pieChartComponent = null;
      }

    } else if (this.state.resultFinishedLoading === false) {
        predictionResult = <Loader 
                            className="PredictResult"
                            type="Rings"
                            color="#3f51b5"
                            height={150}
                            width={150}
                            />;
        pieChartComponent = null;
    } else {
        predictionResult = '';
        pieChartComponent = null;
    }

    return (
        <div className="Main">
            {/* <p className="Title" style={{fontFamily: 'sans-serif', fontSize: 45, color: '#696969'}}>RECTnet</p> */}
            <div className="MainButtons"> 
              <Button className="UploadButton" variant="contained" color="primary" component="label" style={{marginRight: '5px', marginBottom: '5px'}}>
                Upload
                <input type="file" onChange={this.fileSelectedHandler} style={{ display: "none" }}/>
              </Button> 
              <Button className="PredictButton" onClick={this.predictHandler} variant="contained" color="primary" component="span" style={{marginRight: '5px', marginBottom: '5px'}}>
                Predict
              </Button>
            </div> 
            <div>
              {predictionResult}
            </div>
            <Grid className="Image" container spacing={1} >
              <Grid item xs={9} >
                <img className="ImageBox" src={this.state.imagePreviewUrl} style={{...imageStyle}} />
              </Grid>
              <Grid item xs={1}>
                {pieChartComponent}
              </Grid>
            </Grid>
        </div>
    );
  }
}

export default Main;