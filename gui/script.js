
// update rate
var tick = 500;
var autorefresh = true;

// show error in document
function showErr(err){
	document.getElementById("ErrorBox").innerHTML = err;
}

function setautorefresh(){
	autorefresh =  document.getElementById("AutoRefresh").checked;
}

// refreshes the contents of all dynamic elements,
// leaves the rest of the page alone.
function refresh(){
	if (autorefresh){
		showErr("");
		var response;
		try{
			var req = new XMLHttpRequest();
			req.open("POST", "/refresh/", false);
			req.timeout = tick;
			req.send(null);
			response = JSON.parse(req.responseText);	
			for(var i=0; i<response.length; i++){
				var r = response[i];
				document.getElementById(r.ID).innerHTML = r.HTML;
			}
		}catch(e){
			showErr(e);
		}
	}
}

function rpc(method){
	try{
		var req = new XMLHttpRequest();
		req.open("POST", "/rpc/", false);
		req.timeout = tick;
		var map = {"Method": method};
		req.send(JSON.stringify(map));
	}catch(e){
		showErr(e);
	}
	refresh();
}

setInterval(refresh, tick);