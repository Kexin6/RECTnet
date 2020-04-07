import React, { Component } from 'react';
import { Container, Grid, Button, Paper, ListItem, List, CHeckbox, ListItemIcon, ListItemText,  ListItemSecondaryAction, IconButton, Box, Divider} from '@material-ui/core';
import TimelineIcon from '@material-ui/icons/Timeline';
import DeleteForeverIcon from '@material-ui/icons/DeleteForever';
import { PureComponent } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

const emotion_classes = ['Neutral',
    'Happiness',        
    'Sadness',   
    'Surprise',  
    'Fear',      
    'Disgust',   
    'Anger']      

const emotion_colors = [
    '#888888',
    '#4AB847',
    '#1962AF',
    '#FCE826',
    '#F69120',
    '#612D91',
    '#E71D25'
]

const dateOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
class HistoryPreview extends Component {
    constructor(props) {
        super(props);
        this.state = { 
            data: [],
            showFullHistory: false,
            chartEmotions: [true, true, false, false, false, false, false],
        };
    }

    updateHistory = () => {
        var data = window.localStorage.getItem('emo-history');
        if (!!data) this.setState({data: JSON.parse(data)})
        console.log("Preview Component loaded")
        // console.log(JSON.parse(data))
    }

    setShowFullHistory = (show=true) => {
        this.setState({showFullHistory: show})
    }

    setChartEmotions = id => {
        let chartEmotions = this.state.chartEmotions
        chartEmotions[id] = !chartEmotions[id]
        this.setState({chartEmotions})
    }

    updateDimensions = () => {
        this.setState({
            dimensions: {
              width: window.innerWidth,
              height: window.innerHeight,
            },
          });
    };

    componentDidMount() {
        this.updateHistory()
        this.setState({
            dimensions: {
              width: window.innerWidth,
              height: window.innerHeight,
            },
        });
        window.addEventListener('resize', this.updateDimensions);
    }

    componentDidUpdate(prevProps) {
        if (this.props.hisCount !== prevProps.hisCount) this.updateHistory()
    }

    render() {
        var display;
        var hData = this.state.data;
        var styles = {
            emptyLine: {
                marginBottom:"20px"
            },
            margin: {
                margin: "10px"
            }
        }

        if (!this.state.data || this.state.data.length === 0) {
            display = <h3>No history</h3>
        } else {
            // console.log(typeof (hData[hData.length-1].time))
            display = <h3>
                Last time used {new Date(hData[hData.length-1].time).toLocaleDateString("en-US", dateOptions)}
                </h3>
        }

        if (!this.state.showFullHistory) {
            var historyButton = <Button disabled={!this.state.data} 
                onClick={()=>{this.setShowFullHistory(true)}}>
                <TimelineIcon/> &nbsp; Show full history
            </Button>
        } else {
            var historyButton = <Button 
                onClick={()=>{this.setShowFullHistory(false)}}>
                <TimelineIcon/> &nbsp; Hide full history
            </Button>
        }

        var historyPanel = <React.Fragment>Graph</React.Fragment>
        if (this.state.showFullHistory) {

            var displayData = []
            var trendData = []
            hData.forEach((v, i) => {
                let {data, time} = v
                time = new Date(time)
                let maxIndex = data.indexOf(Math.max.apply(Math, data))
                displayData.push(
                    {time: time.toLocaleString(),
                    emotion: emotion_classes[maxIndex],
                    confidence: data[maxIndex]}
                )

                // for trend graph
                trendData.push(
                    {
                        name: time.toLocaleString(),
                        [emotion_classes[0]]: data[0],
                        [emotion_classes[1]]: data[1],
                        [emotion_classes[2]]: data[2],
                        [emotion_classes[3]]: data[3],
                        [emotion_classes[4]]: data[4],
                        [emotion_classes[5]]: data[5],
                        [emotion_classes[6]]: data[6]
                    }
                )
            })

            var historyList;
            if (this.state.dimensions.width > 650) {
                historyList = <List m={2}>
                    {
                        displayData.map((v,i) => {
                            return <ListItem key={i} role={undefined} dense button onClick={()=>{this.setState()}} >
                                <ListItemText id={i} primary={`Mood: ${v.emotion}, confidence: ${v.confidence}%`} />
                                <ListItemSecondaryAction>
                                    <Grid container spacing={2} alignItems="center">
                                        <ListItemText id={i} primary={`Saved at ${v.time}`} />
                                        <IconButton aria-label="delete" className={emotion_classes.margin}>
                                            <DeleteForeverIcon fontSize="small" />
                                        </IconButton>
                                    </Grid>
                                </ListItemSecondaryAction>
                            </ListItem>
                        })
                    }
                </List>   
            } else {
                historyList = <Grid alignItems="center"> 
                    <List m={2}>
                        {
                            displayData.map((v,i) => {
                                return <ListItem key={i} role={undefined} dense button onClick={()=>{this.setState()}} divider={true}>                                      
                                        <Box display="flex" flexDirection="row" width={1}>
                                                <Box display="flex" flexDirection="column" flexGrow={1}>
                                                    <Box m="0">
                                                        <ListItemText id={i} primary={`${v.emotion}: ${v.confidence}%`} />
                                                    </Box>
                                                    <Box m="0">                                            
                                                        <ListItemText id={i} primary={`${v.time}`} />                                              
                                                    </Box>
                                                </Box>
                                                <Box alignItems="flex-end">
                                                    <IconButton aria-label="delete" className={emotion_classes.margin}>
                                                        <DeleteForeverIcon fontSize="medium" />
                                                    </IconButton>
                                                </Box>   
                                            </Box>
                                               
                                        </ListItem>                                                                 
                            }) 
                        }                
                    </List>   
                </Grid>
            }   
                  
            var displayList = []
            this.state.chartEmotions.forEach((v,i)=>{
                if (v === true)
                    displayList.push(emotion_classes[i])
            })

            var trendGraph = 
                    <LineChart
                        width={this.state.dimensions.width / 1.2 - 40}
                        height={300}
                        data={trendData}
                        margin={{
                        top: 5, right: 10, left: 10, bottom: 5,
                        }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {
                            displayList.map((v,i)=> <Line key={i} type="monotone" dataKey={v} 
                                stroke={emotion_colors[emotion_classes.indexOf(v)]}/>)
                        }
                    </LineChart>

            historyPanel = <React.Fragment>
                <Grid container spacing={2} direction="column" direction="column">
                    {historyList}
                    <Divider style={styles.emptyLine}/>
                </Grid>
                <Grid container spacing={2} direction="row" justify="center">
                    {emotion_classes.map((emotion,i)=>{
                        let color = this.state.chartEmotions[i] ? "primary" : "default"
                        let variant = this.state.chartEmotions[i] ? "contained" : "outlined"
                        return <Button size="small" key={i} onClick={()=>this.setChartEmotions(i)} color={color} variant={variant} style={styles.margin}>
                            {emotion}
                        </Button>
                    })}
                </Grid>
                {trendGraph}

                <Divider style={styles.emptyLine}/>
            </React.Fragment>
        }
        
        return (
            <div>
                <Container maxWidth="lg">
                    <Grid container spacing={2} >
                        <Grid container spacing={2} justify='space-between'>
                            {display}
                            {historyButton}
                        </Grid>
                        {this.state.showFullHistory && historyPanel}
                    </Grid>
                </Container>
            </div>
        );
    }
}

export default HistoryPreview;