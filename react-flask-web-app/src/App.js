import React, { Component } from 'react'
import { BrowserRouter as Router, Route } from 'react-router-dom'
import axios from 'axios';

import Navbar from './components/Navbar'
import Landing from './components/Landing'
import Login from './components/Login'
import Register from './components/Register'
import Profile from './components/Profile'
import Cam from './components/Cam'

class App extends Component {
  state = {
    selectedFile: null
  }

  fileSelectedHandler = event => {
    this.setState({
        selectedFile: event.target.files[0]
    })
  }

  fileUploadHandler = () => {
    const fd = new FormData();
    fd.append('image', this.state.selectedFile, this.state.selectedFile.name);
    axios.post('', fd, {
          onUploadProgress: progressEvent => {
            console.log('Upload Progress: ' + Math.round(progressEvent.loaded/progressEvent.total*100) + '%')
          }
        } )
        .then(res => {
            console.log(res);
        });
  }

  render() {
    return (
      <Router>
        <div className="App">
          <Navbar />
          <Route exact path="/" component={Landing} />
          <Route exact path="/" component={Cam} />
          <div className="fileUpload">
              <input
                type="file"
                onChange={this.fileSelectedHandler}
                ref={fileInput => this.fileInput = fileInput}/>
              <button onClick={this.fileUploadHandler}>Upload</button>
          </div>
          <div className="container">
            <Route exact path="/register" component={Register} />
            <Route exact path="/login" component={Login} />
            <Route exact path="/profile" component={Profile} />
          </div>
        </div>
      </Router>
    )
  }
}
//              <button onClick={() => this.fileInput.click()}>Pick File</button>

export default App
