<!DOCTYPE html>
<html>
  <head>
    <title>This link is blocked</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <link href="/src/css/bootstrap.min.css" rel="stylesheet" />
    <script src="/src/js/jquery.min.js"></script>
    <script src="/src/js/popper.min.js"></script>
    <script src="/src/js/bootstrap.min.js"></script>
    <script src="/src/js/locale.js"></script>
    <link href="/src/css/layout.css" rel="stylesheet" />
    <link href="/src/fonts/font-awesome.min.css" rel="stylesheet" />
    <meta name="keywords" content="" />
    <meta name="description" content="" />
  </head>
  <body>
    <div id="header">
      <div class="header-image-container">
        <img class="header-image" src="/src/img/campus_logo_left.png" alt="Header logo" />
      </div>
      <div class="float-right p-3">
        <a class="btn btn-primary" id="langtoggle">
          <span lang="en">Deutsch</span>
          <span lang="de">English</span>
        </a>
      </div>
    </div>
    <div class="container text-center mb-5 pb-5" id="content">
      <h2>
        <i class="fa fa-lock mr-1 text-danger p-3" aria-hidden="true"></i>
        <span lang="en">This link is blocked</span>
        <span lang="de">Dieser Link ist gesperrt</span>
      </h2>

      <div id="badurl" class="alert alert-danger" role="alert">
      </div>

      <div id="sslignored" class="alert alert-warning">
	<span lang="en">Also, in order to get here, you seem to have ignored an SSL certificate warning</span>
	<span lang="de">Um hierher zu gelangen haben Sie außerdem eine Zertifikatswarnung ignoriert.</span>
      </div>

      <p>
        <span lang="en">You are seeing this page because you opened a link which was blocked by our campus DNS firewall. The link is blocked, because it is known to be used for malicious purposes. It will likely not lead where you expect, but instead try to trick you into downloading malware or disclosing your password.</span>
        <span lang="de">Dieser Link wurde von unserer Campus DNS Firewall gesperrt, weil er mutmaßlich f&uuml;r b&ouml;swillige Zwecke verwendet wird. Sehr wahrscheinlich wird der Link nicht zum erwarteten Ziel f&uuml;hren und Sie stattdessen zur Installation von Malware oder der Preisgabe Ihres Passworts verleiten.</span>
      </p>
      <p>
        <span lang="en">In order to prevent damage to your computer and our campus network, we are trying to block bad links, while leaving normal network traffic unaffected.</span>
        <span lang="de">Wir versuchen, schadhafte Links zu blockieren, um negative Folgen f&uuml;r Ihren Computer und unser Campus-Netzwerk zu verhindern. Normaler Netzwerkverkehr wird dadurch nicht beeintr&auml;chtigt.</span>
      </p>
      <p>
        <span lang="en">Please note that we will not record your IP address, nor will this have any consequences for you personally. The block lists we use are just for stopping malware and computer viruses, not to impose any form of censorship.</span>
        <span lang="de">Ihre IP-Adresse wird weder aufgezeichnet noch wird dieses Ereignis irgendwelche Konsequenzen f&uuml;r Sie pers&ouml;nlich haben. Unsere Blocklisten sind nur daf&uuml;r da, Malware und Computerviren zu stoppen, nicht aber zu Zwecken der Zensur.</span>
      </p>
      <h3>
        <span lang="en">What should I do now?</span>
        <span lang="de">Was muss ich jetzt tun?</span>
      </h3>
      <p>
        <span lang="en">You do not have to do anything. However, if you think the link is genuine and we made a mistake, please contact your institute IT service. Also, we welcome information about how you got to that bad link: was this a spam/phishing email or some other source? Your information can help us protect our services, so please get in touch.</span>
        <span lang="de">Sie m&uuml;ssen nichts tun. Wenn Sie jedoch der Ansicht sind, dass unsere Firewall im Irrtum ist und der fragliche Link ungef&auml;hrlich ist, dann kontaktieren Sie bitte Ihre Instituts-IT. Ebenso w&uuml;rden wir es sehr begr&uuml;&szlig;en, wenn Sie uns mitteilen, auf welche Weise Sie an diesen Link gekommen sind: Zum Beispiel &uuml;ber Spam in Ihrem E-Mail-Konto oder auf anderem Weg. Ihre Angaben k&ouml;nnten uns helfen, den Dienst zu verbessern.</span>
      </p>
      <small>
        <address>
          <span lang="en">The IT departments of the T&uuml;bingen Max Planck Institutes.
          <br />To avoid misinterpretation, this page does not contain any clickable links or addresses.</span>
          <span lang="de">Die IT Abteilungen der T&uuml;binger Max-Planck-Institute
          <br />Um Missverst&auml;ndnisse zu vermeiden enh&auml;lt diese Seite keine klickbaren Links oder Adressen.</span>
        </address>
      </small>
    </div>
    <br />
    <div id="footer-bottom">
      <p class="float-left pt-4 privacy">
        <span lang="en">© 2021 Max Planck Society</span>
        <span lang="de">© 2021 Max-Planck-Gesellschaft</span>
      </p>
      <p class="float-left pt-4 privacy">
        <span lang="en">Page last edited June 9 2021</span>
        <span lang="de">Letzte Änderung: 9. Juni 2021</span>
      </p>
    </div>
    <script>
         var badurl = document.URL;
	 document.getElementById("badurl").innerHTML = badurl;
    </script>
    <!-- Matomo -->
    <script type="text/javascript">
  var _paq = window._paq || [];
    /* tracker methods like "setCustomDimension" should be called before "trackPageView" */
    _paq.push(['trackPageView']);
      _paq.push(['enableLinkTracking']);
        (function() {
		    var u="https://piwik.tuebingen.mpg.de/";
		        _paq.push(['setTrackerUrl', u+'matomo.php']);
			    _paq.push(['setSiteId', '44']);
			        var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
				    g.type='text/javascript'; g.async=true; g.defer=true; g.src=u+'matomo.js'; s.parentNode.insertBefore(g,s);
				      })();
    </script>
    <!-- End Matomo Code -->
  </body>
</html>
