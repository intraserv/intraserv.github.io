var terminalContainer = document.getElementById('terminal-container'),
    term = new Terminal(),
    socket,
    termid;
 
term.open(terminalContainer);
term.fit();
 
if (document.location.pathname) {
  var parts = document.location.pathname.split('/')
    , base = parts.slice(0, parts.length - 1).join('/') + '/'
    , resource = base.substring(1) + 'socket.io';
 
  socket = io.connect(null, { resource: resource });
} else {
  socket = io.connect();
}
 
var cols=80,
    rows=24;                  
 
socket.emit('create', cols, rows, function(err, data) {
  if (err) return self._destroy();
  self.pty = data.pty;
  self.id = data.id;
  termid = self.id;
  term.emit('open tab', self);
});
 
term.writeln('Welcome to xterm.js');
term.writeln('Connecting to websocket...');
 
term.on('data', function(data) {
  socket.emit('data', termid, data);
});
 
term.on('resize', function(data) {
  socket.emit('resize', termid, term.cols, term.rows);
});
 
socket.on('connect', function() {
  term.writeln('Connected.');
  term.writeln('');
});
 
socket.on('data', function(id, data) {
  term.write(data);
});