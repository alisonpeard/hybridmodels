<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis minScale="1e+08" version="3.20.3-Odense" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" mode="0" fetchMode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option type="bool" name="WMSBackgroundLayer" value="false"/>
      <Option type="bool" name="WMSPublishDataSourceUrl" value="false"/>
      <Option type="int" name="embeddedWidgets/count" value="0"/>
      <Option type="QString" name="identify/format" value="Value"/>
    </Option>
  </customproperties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer type="singlebandpseudocolor" nodataColor="" alphaBand="-1" classificationMax="12" classificationMin="0" opacity="1" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader classificationMode="1" maximumValue="12" clip="0" minimumValue="0" colorRampType="INTERPOLATED" labelPrecision="0">
          <colorramp type="gradient" name="[source]">
            <Option type="Map">
              <Option type="QString" name="color1" value="255,255,212,255"/>
              <Option type="QString" name="color2" value="153,52,4,255"/>
              <Option type="QString" name="discrete" value="0"/>
              <Option type="QString" name="rampType" value="gradient"/>
              <Option type="QString" name="stops" value="0.25;254,217,142,255:0.5;254,153,41,255:0.75;217,95,14,255"/>
            </Option>
            <prop k="color1" v="255,255,212,255"/>
            <prop k="color2" v="153,52,4,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.25;254,217,142,255:0.5;254,153,41,255:0.75;217,95,14,255"/>
          </colorramp>
          <item alpha="255" color="#ffffd4" label="0" value="0"/>
          <item alpha="255" color="#feebb0" label="2" value="1.56"/>
          <item alpha="255" color="#fed68a" label="3" value="3.12"/>
          <item alpha="255" color="#feb555" label="5" value="4.68"/>
          <item alpha="255" color="#fb9427" label="6" value="6.24"/>
          <item alpha="255" color="#e87619" label="8" value="7.800000000000001"/>
          <item alpha="255" color="#d15a0d" label="9" value="9.36"/>
          <item alpha="255" color="#b34508" label="11" value="10.8"/>
          <item alpha="255" color="#993404" label="12" value="12"/>
          <rampLegendSettings orientation="2" minimumLabel="" suffix="" direction="0" prefix="" useContinuousLegend="1" maximumLabel="">
            <numericFormat id="basic">
              <Option type="Map">
                <Option type="QChar" name="decimal_separator" value=""/>
                <Option type="int" name="decimals" value="6"/>
                <Option type="int" name="rounding_type" value="0"/>
                <Option type="bool" name="show_plus" value="false"/>
                <Option type="bool" name="show_thousand_separator" value="true"/>
                <Option type="bool" name="show_trailing_zeros" value="false"/>
                <Option type="QChar" name="thousand_separator" value=""/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast contrast="0" gamma="1" brightness="0"/>
    <huesaturation saturation="0" colorizeBlue="128" colorizeGreen="128" grayscaleMode="0" colorizeOn="0" colorizeRed="255" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
