const express = require('express')

const app = express()

app.use(express.json())

app.post('/saveOnMongo', (req, res) => {
  console.log(req.body);
  res.send()
})

app.listen(1997, () => { console.log('Running'); })